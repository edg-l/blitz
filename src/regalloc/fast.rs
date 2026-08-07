//! Linear-scan allocation, for `-O0`.
//!
//! The colouring allocator in `global_allocator` is the one that produces good
//! code, and it is the only one `-O1` uses. This one exists for three reasons
//! that have nothing to do with the code it emits:
//!
//! 1. **It is a second implementation.** `run_diff.sh`'s `-O0`-vs-`-O1` oracle
//!    is blind to anything equally wrong at both levels, so while both levels
//!    shared one allocator the component the bug priors rank first was the one
//!    that comparison could not see. Two allocators make an allocation bug a
//!    disagreement rather than a shared answer.
//! 2. **It cannot refuse.** Colouring fails when the chromatic number exceeds
//!    the register file and spilling cannot bring it down; a program that does
//!    not compile is a program no oracle can judge. A scan spills whatever does
//!    not fit and always finishes, so every program gets an answer at one level.
//! 3. **Locals in frame slots is what debug info describes**, which is the
//!    shape `-O0` wants once DWARF exists.
//!
//! **This is a first attempt and it does not work; see ROADMAP item 1.** It
//! allocates whole live ranges, which forces a value live in an early block and
//! a late one to hold a register across everything between. Block parameters
//! cannot be spilled, so a loop-heavy function meets a wall of them with nothing
//! the scan may take: on `tests/lit/bench/sieve.c` that is round 0, before any
//! spilling. Pre-spilling the long values first was tried and is worse, because
//! a spill turns a spillable value into reloads and a reload cannot be spilled
//! in turn. The model that does not have this failure holds nothing across an
//! instruction, so live ranges stop mattering -- that is the rewrite.
//!
//! What it deliberately does not do: no interference graph, no coalescing, no
//! live-range splitting, no rematerialization. It is linear in instructions
//! after liveness.

use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::CALLEE_SAVED;
use crate::x86::reg::{Reg, RegClass};

use super::coloring::{allocatable_gpr_order, allocatable_xmm_order};
use super::global_liveness::compute_global_liveness_with_block_params;
use super::interference::VRegSet;
use super::slots::SlotAllocator;
use super::{GlobalRegAllocResult, build_vreg_classes_from_all_blocks};

/// A round that spills nothing new has converged; one that does not is retried.
/// Each round strictly shortens the intervals it spilled, so this bound is a
/// backstop against a bug rather than an expected limit.
const MAX_SPILL_ROUNDS: usize = 16;

/// Where one VReg is live, as a half-open span of the flattened instruction
/// numbering. Blocks are laid end to end, so a VReg live across a block
/// boundary spans everything between -- the scan is conservative about holes it
/// cannot see, which costs registers and never correctness.
#[derive(Clone, Copy, Debug)]
struct Interval {
    vreg: VReg,
    start: usize,
    end: usize,
}

/// Linear-scan allocation over `block_schedules`.
///
/// Returns the same shape as `allocate_global`, with `coalesce_aliases` always
/// empty: nothing here merges VRegs, so no operand needs renaming.
#[allow(clippy::too_many_arguments)]
pub fn allocate_fast(
    block_schedules: &[Vec<ScheduledInst>],
    param_vregs: &[(VReg, Reg)],
    call_arg_precolors: Vec<(VReg, Reg)>,
    phi_uses: &[VRegSet],
    cfg_succs: &[Vec<usize>],
    block_param_vregs_per_block: &[VRegSet],
    func_name: &str,
    uses_frame_pointer: bool,
    slots: &mut SlotAllocator,
) -> Result<GlobalRegAllocResult, String> {
    let mut schedules: Vec<Vec<ScheduledInst>> = block_schedules.to_vec();
    let mut next_vreg: u32 = schedules
        .iter()
        .flatten()
        .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    // Precolored VRegs hold a register the ABI named, so the scan may not move
    // them and may not spill them. A call argument must be in its argument
    // register at the call; a parameter arrives in one.
    let mut precolored: BTreeMap<VReg, Reg> = BTreeMap::new();
    for &(v, r) in param_vregs {
        precolored.insert(v, r);
    }
    for (v, r) in call_arg_precolors {
        precolored.insert(v, r);
    }

    let gpr_order = allocatable_gpr_order(uses_frame_pointer);
    let xmm_order = allocatable_xmm_order();
    let callee_saved: BTreeSet<Reg> = CALLEE_SAVED.iter().copied().collect();

    // Every VReg from here up is one a spill round minted: a reload, defined
    // immediately before the use it feeds. Spilling one stores a value that was
    // just loaded and mints another reload to load it again, which is a cascade
    // with no fixpoint -- 40 of 55 refusals were this, and raising the round
    // limit to 64 moved one program.
    let first_reload_vreg = next_vreg;

    let mut spilled: BTreeSet<usize> = BTreeSet::new();

    for round in 0..MAX_SPILL_ROUNDS {
        let vreg_classes = build_vreg_classes_from_all_blocks(&schedules);
        let liveness = compute_global_liveness_with_block_params(
            &schedules,
            cfg_succs,
            phi_uses,
            block_param_vregs_per_block,
        );

        let layout = BlockLayout::new(&schedules);
        let intervals = build_intervals(&schedules, &liveness, &layout, &vreg_classes);
        let (call_points, div_points) = clobber_positions(&schedules, &layout);

        // What no store can move, so the scan must never choose it as a victim:
        // a block parameter is written by a phi copy on the edge and only the
        // splitter's slot routing can relieve one, a result the hardware pins is
        // in its register by the instruction's own definition, and a call
        // argument has to be in its argument register at the call. Demanding one
        // of these produces a spill that changes nothing and a round that reads
        // as "did not converge".
        let mut unspillable: BTreeSet<VReg> = precolored.keys().copied().collect();
        for params in block_param_vregs_per_block {
            for v in params.iter() {
                unspillable.insert(VReg(v as u32));
            }
        }
        for inst in schedules.iter().flatten() {
            if inst.op.result_in_fixed_regs() {
                unspillable.insert(inst.dst);
            }
        }
        for insts in &schedules {
            for v in super::spill::collect_call_arg_vregs(insts) {
                unspillable.insert(VReg(v as u32));
            }
        }
        for insts in &schedules {
            for inst in insts {
                for v in std::iter::once(inst.dst).chain(inst.operands.iter().copied()) {
                    if v.0 >= first_reload_vreg {
                        unspillable.insert(v);
                    }
                }
            }
        }

        let scan = Scan {
            gpr_order: &gpr_order,
            xmm_order: &xmm_order,
            callee_saved: &callee_saved,
            precolored: &precolored,
            vreg_classes: &vreg_classes,
            call_points: &call_points,
            div_points: &div_points,
            unspillable: &unspillable,
        };
        let outcome = scan.run(&intervals);

        if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
            eprintln!(
                "[regalloc] fast {func_name} round {round}: {} intervals, {} unplaced, {} spilled so far",
                intervals.len(),
                outcome.to_spill.len(),
                spilled.len()
            );
        }

        if outcome.to_spill.is_empty() {
            if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
                eprintln!(
                    "[regalloc] fast {func_name}: {} vregs assigned after {} spill round(s), {} spilled",
                    outcome.assignment.len(),
                    round,
                    spilled.len()
                );
            }
            return Ok(finish(
                schedules,
                outcome.assignment,
                &callee_saved,
                param_vregs,
            ));
        }

        if !outcome.unplaceable.is_empty() {
            let names: Vec<String> = outcome
                .unplaceable
                .iter()
                .take(8)
                .map(|v| format!("v{}", v.0))
                .collect();
            return Err(format!(
                "fast regalloc: {} value(s) need a register that nothing can give up \
                 [{}] -- every value live there is a block parameter, a call argument, \
                 a result the hardware pins, or a reload (in function '{func_name}')",
                outcome.unplaceable.len(),
                names.join(", ")
            ));
        }

        let new_spills: BTreeSet<usize> = outcome
            .to_spill
            .iter()
            .copied()
            .filter(|v| !spilled.contains(v))
            .collect();

        // Nothing new to spill and still not placed: the pressure point is one
        // instruction whose own operands are what is live there, which spilling
        // cannot relieve -- the same wall the colouring allocator reports.
        if new_spills.is_empty() {
            return Err(format!(
                "fast regalloc: {} value(s) could not be placed and none is spillable; \
                 the pressure point is one instruction whose own operands are what is \
                 live there (in function '{func_name}')",
                outcome.to_spill.len()
            ));
        }

        // Only what this round chose, against schedules that already carry every
        // earlier round's spill code. Handing over the accumulated set would
        // spill each of those a second time, and the reloads of a reload are
        // what a "did not converge" looks like from outside.
        spilled.extend(new_spills.iter().copied());
        super::spill::insert_spills_global(
            &mut schedules,
            &new_spills,
            slots,
            &mut next_vreg,
            &vreg_classes,
        );
    }

    Err(format!(
        "fast regalloc: did not converge in {MAX_SPILL_ROUNDS} spill rounds \
         (in function '{func_name}')"
    ))
}

/// Where each block starts in the flattened numbering.
struct BlockLayout {
    /// `base[b]` is the position of block `b`'s first instruction.
    base: Vec<usize>,
}

impl BlockLayout {
    fn new(schedules: &[Vec<ScheduledInst>]) -> Self {
        let mut base = Vec::with_capacity(schedules.len());
        let mut pos = 0usize;
        for insts in schedules {
            base.push(pos);
            // One extra position per block, so a value live out of a block ends
            // strictly after its last instruction and a value live in to the
            // next starts strictly before that block's first.
            pos += insts.len() + 1;
        }
        BlockLayout { base }
    }

    fn pos(&self, block: usize, inst: usize) -> usize {
        self.base[block] + inst
    }

    fn block_end(&self, block: usize, len: usize) -> usize {
        self.base[block] + len
    }
}

/// The live span of every VReg that needs a register.
///
/// Flags-class values are left out entirely: EFLAGS is not in any register file
/// the scan allocates from, and the colouring allocator gives them no machine
/// register either.
fn build_intervals(
    schedules: &[Vec<ScheduledInst>],
    liveness: &super::global_liveness::GlobalLiveness,
    layout: &BlockLayout,
    vreg_classes: &BTreeMap<VReg, RegClass>,
) -> Vec<Interval> {
    let mut span: BTreeMap<VReg, (usize, usize)> = BTreeMap::new();

    let touch = |span: &mut BTreeMap<VReg, (usize, usize)>, v: VReg, at: usize| {
        if matches!(vreg_classes.get(&v), Some(RegClass::Flags)) {
            return;
        }
        span.entry(v)
            .and_modify(|e| {
                e.0 = e.0.min(at);
                e.1 = e.1.max(at);
            })
            .or_insert((at, at));
    };

    for (b, insts) in schedules.iter().enumerate() {
        let entry = layout.base[b];
        let exit = layout.block_end(b, insts.len());

        for v in liveness.live_in[b].iter() {
            touch(&mut span, VReg(v as u32), entry);
        }
        for v in liveness.live_out[b].iter() {
            touch(&mut span, VReg(v as u32), exit);
        }

        for (i, inst) in insts.iter().enumerate() {
            let at = layout.pos(b, i);
            for &op in &inst.operands {
                touch(&mut span, op, at);
            }
            if !inst.op.has_no_result() {
                touch(&mut span, inst.dst, at);
            }
        }
    }

    let mut intervals: Vec<Interval> = span
        .into_iter()
        .map(|(vreg, (start, end))| Interval { vreg, start, end })
        .collect();
    intervals.sort_by_key(|i| (i.start, i.end, i.vreg.0));
    intervals
}

/// Flattened positions of the two instruction kinds that take registers out of
/// the scan's hands: a call clobbers everything caller-saved, and a division
/// reads RAX and RDX as its dividend whatever its operands wanted.
fn clobber_positions(
    schedules: &[Vec<ScheduledInst>],
    layout: &BlockLayout,
) -> (Vec<usize>, Vec<usize>) {
    use crate::ir::op::{MachOp, Op, PseudoOp};
    let mut calls = Vec::new();
    let mut divs = Vec::new();
    for (b, insts) in schedules.iter().enumerate() {
        for (i, inst) in insts.iter().enumerate() {
            match inst.op {
                Op::Pseudo(PseudoOp::CallResult(..)) | Op::Pseudo(PseudoOp::VoidCallBarrier) => {
                    calls.push(layout.pos(b, i))
                }
                Op::Mach(MachOp::X86Idiv(..)) | Op::Mach(MachOp::X86Div(..)) => {
                    divs.push(layout.pos(b, i))
                }
                _ => {}
            }
        }
    }
    calls.sort_unstable();
    divs.sort_unstable();
    (calls, divs)
}

struct Scan<'a> {
    gpr_order: &'a [Reg],
    xmm_order: &'a [Reg],
    callee_saved: &'a BTreeSet<Reg>,
    precolored: &'a BTreeMap<VReg, Reg>,
    vreg_classes: &'a BTreeMap<VReg, RegClass>,
    call_points: &'a [usize],
    div_points: &'a [usize],
    unspillable: &'a BTreeSet<VReg>,
}

struct ScanOutcome {
    assignment: BTreeMap<VReg, Reg>,
    to_spill: Vec<usize>,
    /// Values that got no register and that no store can move. This is the wall
    /// itself, not a round that failed to make progress.
    unplaceable: Vec<VReg>,
}

impl Scan<'_> {
    /// Must this value still be in its register after a call? Strictly after
    /// its end is what matters -- a value whose last use *is* the call does not
    /// have to survive it. Every XMM is caller-saved, so an XMM that does have
    /// to survive one can only be spilled.
    fn crosses_call(&self, iv: &Interval) -> bool {
        self.call_points.iter().any(|&c| c > iv.start && c < iv.end)
    }

    /// Is this value live anywhere a division reads RAX and RDX as its
    /// dividend? A divisor sitting in either is the one case lowering asserts
    /// against, and unlike a call the constraint binds *at* the instruction, so
    /// a value whose last use is the division is still subject to it.
    fn meets_div(&self, iv: &Interval) -> bool {
        self.div_points
            .iter()
            .any(|&d| d >= iv.start && d <= iv.end)
    }

    fn class_of(&self, v: VReg) -> RegClass {
        self.vreg_classes.get(&v).copied().unwrap_or(RegClass::GPR)
    }

    fn run(&self, intervals: &[Interval]) -> ScanOutcome {
        let mut assignment: BTreeMap<VReg, Reg> = BTreeMap::new();
        let mut to_spill: Vec<usize> = Vec::new();
        let mut unplaceable: Vec<VReg> = Vec::new();
        // Active intervals, each with the register it holds, kept sorted by end
        // so expiry is a prefix.
        let mut active: Vec<(usize, Reg, VReg)> = Vec::new();

        for iv in intervals {
            active.retain(|&(end, _, _)| end >= iv.start);

            let class = self.class_of(iv.vreg);
            let held: BTreeSet<Reg> = active.iter().map(|&(_, r, _)| r).collect();

            if let Some(&fixed) = self.precolored.get(&iv.vreg) {
                // The ABI named this register, so whoever else holds it moves.
                for &(_, r, v) in &active {
                    if r == fixed && self.precolored.get(&v) != Some(&fixed) {
                        to_spill.push(v.0 as usize);
                    }
                }
                active.retain(|&(_, r, _)| r != fixed);
                assignment.insert(iv.vreg, fixed);
                active.push((iv.end, fixed, iv.vreg));
                active.sort_by_key(|&(end, _, _)| end);
                continue;
            }

            let crosses = self.crosses_call(iv);
            let pool: &[Reg] = match class {
                RegClass::GPR => self.gpr_order,
                RegClass::XMM => self.xmm_order,
                // Filtered out when the intervals were built.
                RegClass::Flags => continue,
            };

            // Every XMM is caller-saved, so one whose value has to survive a
            // call cannot stay in a register at all.
            if crosses && class == RegClass::XMM && !self.unspillable.contains(&iv.vreg) {
                to_spill.push(iv.vreg.0 as usize);
                continue;
            }

            let div_bound = class == RegClass::GPR && self.meets_div(iv);
            let choice = pool.iter().copied().find(|r| {
                !held.contains(r)
                    && (!crosses || self.callee_saved.contains(r))
                    && (!div_bound || (*r != Reg::RAX && *r != Reg::RDX))
            });

            match choice {
                Some(r) => {
                    assignment.insert(iv.vreg, r);
                    active.push((iv.end, r, iv.vreg));
                    active.sort_by_key(|&(end, _, _)| end);
                }
                None => {
                    // The furthest-ending active value this scan is allowed to
                    // move. A block parameter, a pinned result and a call
                    // argument are all excluded: spilling one changes nothing,
                    // and the round that follows reads as a failure to converge
                    // rather than as the wall it is.
                    let victim = active
                        .iter()
                        .filter(|&&(_, r, v)| {
                            self.class_of(v) == class
                                && !self.unspillable.contains(&v)
                                && pool.contains(&r)
                        })
                        .max_by_key(|&&(end, _, _)| end)
                        .copied();

                    // Evict when the victim outlives this interval, and also
                    // whenever this one cannot be spilled itself -- then the
                    // register has to come from somewhere, whatever the ends say.
                    let must_place = self.unspillable.contains(&iv.vreg);
                    match victim {
                        Some((end, r, v)) if end > iv.end || must_place => {
                            to_spill.push(v.0 as usize);
                            active.retain(|&(_, _, av)| av != v);
                            assignment.remove(&v);
                            assignment.insert(iv.vreg, r);
                            active.push((iv.end, r, iv.vreg));
                            active.sort_by_key(|&(e, _, _)| e);
                        }
                        // No register and nothing that may give one up. Asking
                        // for this to be spilled is what does not terminate: a
                        // reload spilled mints another reload to load what was
                        // just loaded, and a block parameter spilled is a store
                        // no phi copy reads.
                        _ if must_place => unplaceable.push(iv.vreg),
                        _ => to_spill.push(iv.vreg.0 as usize),
                    }
                }
            }
        }

        to_spill.sort_unstable();
        to_spill.dedup();
        ScanOutcome {
            assignment,
            to_spill,
            unplaceable,
        }
    }
}

fn finish(
    schedules: Vec<Vec<ScheduledInst>>,
    assignment: BTreeMap<VReg, Reg>,
    callee_saved: &BTreeSet<Reg>,
    param_vregs: &[(VReg, Reg)],
) -> GlobalRegAllocResult {
    let mut used: Vec<Reg> = assignment
        .values()
        .copied()
        .filter(|r| callee_saved.contains(r))
        .collect();
    used.sort_by_key(|r| *r as u8);
    used.dedup();

    // A parameter whose precoloring the scan could not keep needs a mov from
    // its ABI register at entry. The scan never drops one, so this is empty --
    // it is reported for the same reason the colouring path reports it, so the
    // caller has one shape to handle.
    let unprecolored_params: Vec<(VReg, Reg)> = param_vregs
        .iter()
        .filter(|(v, r)| assignment.get(v).is_some_and(|got| got != r))
        .copied()
        .collect();

    GlobalRegAllocResult {
        per_block_insts: schedules,
        assignment: assignment
            .into_iter()
            .map(|(v, r)| (v, super::Assignment::Reg(r)))
            .collect(),
        callee_saved_used: used,
        unprecolored_params,
        coalesce_aliases: BTreeMap::new(),
    }
}
