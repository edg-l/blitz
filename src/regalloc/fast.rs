//! Every value in a frame slot, registers borrowed one instruction at a time.
//!
//! The colouring allocator in `global_allocator` is the one that produces good
//! code and the only one `-O1` uses. This one exists for reasons that have
//! nothing to do with the code it emits:
//!
//! 1. **It is a second implementation.** `run_diff.sh`'s `-O0`-vs-`-O1` oracle
//!    is blind to anything equally wrong at both levels, so while both levels
//!    shared one allocator the component the bug priors rank first was the one
//!    that comparison could not see. Two allocators make an allocation bug a
//!    disagreement rather than an answer both levels give.
//! 2. **Locals at fixed frame offsets is what debug info describes**, which is
//!    the shape `-O0` wants once DWARF exists.
//!
//! **The model.** Every value gets a slot. Before each instruction its operands
//! are loaded into fresh VRegs, the instruction writes another fresh VReg, and
//! that is stored back. Nothing is held across an instruction, so no live range
//! is longer than one expansion and the same handful of registers serves the
//! whole function.
//!
//! That is what makes it fit `assignment`, which has one entry per VReg: a
//! *fresh* VReg per use is how one value occupies different registers at
//! different points without the map having to say so. A previous attempt kept
//! whole live ranges and one register each, and could not place a function
//! whose block parameters outnumbered what was left -- see ROADMAP item 1 for
//! what that ruled out.
//!
//! **What it does not do**: no interference graph, no liveness, no coalescing,
//! no splitting, no rematerialization. One pass over the instructions. It does
//! not need the pressure splitter in front of it either, and skipping that is
//! where the speed is: on a 6048-line input `-O0` goes 211.7ms to 150.5ms,
//! 1.41x. Nothing is live across an instruction, so the only shape this cannot
//! place is one instruction reading more operands than the machine has
//! registers -- which no amount of splitting would help.
//!
//! **Not finished.** Forced on it reaches 323 of 334 differential comparisons
//! and refuses 5 programs, against 268 and 55 for the model before it. What is
//! left is one class: an op that *names* a value without writing it into a
//! register. A comparison was the first of them -- `cmp rax,rcx` followed by a
//! store of an unwritten RDX on `tests/lit/asm/rotate.c` -- and asking
//! `produces_flags()` rather than the class map fixed that one. `rotate.c`
//! still returns 0 for 216, so the class has more members; a pair-producing op
//! whose `Proj1` is flags and whose `Proj0` is the value is the place to look
//! next, since the pair VReg itself holds neither.

use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::{Op, PseudoOp, PureOp};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::CALLEE_SAVED;
use crate::x86::reg::{Reg, RegClass};

use super::coloring::{allocatable_gpr_order, allocatable_xmm_order};
use super::interference::VRegSet;
use super::slots::{SlotAllocator, SlotOwner};
use super::{Assignment, GlobalRegAllocResult, build_vreg_classes_from_all_blocks};

/// Where every value of the function lives, and what the scratch VRegs minted
/// to move it resolve to.
struct Frame {
    /// One slot per value. Flags-class VRegs are absent: EFLAGS is in no
    /// register file and no store can reach it.
    slot: BTreeMap<VReg, u32>,
    /// The ABI register a value must occupy where it is read, from function
    /// parameters and call arguments. A load for one of these targets the
    /// register directly instead of a scratch.
    precolored: BTreeMap<VReg, Reg>,
    assignment: BTreeMap<VReg, Assignment>,
    next_vreg: u32,
}

impl Frame {
    fn fresh(&mut self) -> VReg {
        let v = VReg(self.next_vreg);
        self.next_vreg += 1;
        v
    }
}

/// Allocate by giving every value a frame slot.
#[allow(clippy::too_many_arguments)]
pub fn allocate_fast(
    block_schedules: &[Vec<ScheduledInst>],
    param_vregs: &[(VReg, Reg)],
    call_arg_precolors: Vec<(VReg, Reg)>,
    _phi_uses: &[VRegSet],
    _cfg_succs: &[Vec<usize>],
    block_param_vregs_per_block: &[VRegSet],
    func_name: &str,
    uses_frame_pointer: bool,
    slots: &mut SlotAllocator,
) -> Result<GlobalRegAllocResult, String> {
    let classes = build_vreg_classes_from_all_blocks(block_schedules);

    let mut precolored: BTreeMap<VReg, Reg> = BTreeMap::new();
    for &(v, r) in param_vregs {
        precolored.insert(v, r);
    }
    for (v, r) in call_arg_precolors {
        precolored.insert(v, r);
    }

    let next_vreg = block_schedules
        .iter()
        .flatten()
        .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut frame = Frame {
        slot: BTreeMap::new(),
        precolored,
        assignment: BTreeMap::new(),
        next_vreg,
    };

    // Every value that is defined gets a slot, and so does every block
    // parameter -- a parameter is written by the phi copies on its edges rather
    // than by an instruction, so there is no definition in the block to find.
    // Flags are excluded: `RegClass::Flags` is EFLAGS, which no store reaches
    // and which the colouring allocator also leaves without a register.
    for insts in block_schedules {
        for inst in insts {
            if inst.op.has_no_result() || inst.op.produces_flags() || is_flags(&classes, inst.dst) {
                continue;
            }
            claim_slot(&mut frame, slots, inst.dst);
        }
    }
    for params in block_param_vregs_per_block {
        for v in params.iter() {
            let v = VReg(v as u32);
            if !is_flags(&classes, v) {
                claim_slot(&mut frame, slots, v);
            }
        }
    }

    let gpr_pool = allocatable_gpr_order(uses_frame_pointer);
    let xmm_pool = allocatable_xmm_order();

    let mut out: Vec<Vec<ScheduledInst>> = Vec::with_capacity(block_schedules.len());
    for insts in block_schedules {
        let mut block: Vec<ScheduledInst> = Vec::with_capacity(insts.len() * 3);
        for inst in insts {
            expand(
                inst, &mut block, &mut frame, &classes, &gpr_pool, &xmm_pool, func_name,
            )?;
        }
        out.push(block);
    }

    // A block parameter is written by the phi copies on its edges, which
    // `build_phi_copies` turns into a store when the assignment says the
    // parameter lives in a slot. Saying so is what gives it storage at all.
    for params in block_param_vregs_per_block {
        for v in params.iter() {
            let v = VReg(v as u32);
            if let Some(&slot) = frame.slot.get(&v) {
                frame.assignment.insert(v, Assignment::Slot(slot));
            }
        }
    }

    let callee_saved: BTreeSet<Reg> = CALLEE_SAVED.iter().copied().collect();
    let mut used: Vec<Reg> = frame
        .assignment
        .values()
        .filter_map(|a| a.reg())
        .filter(|r| callee_saved.contains(r))
        .collect();
    used.sort_by_key(|r| *r as u8);
    used.dedup();

    if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
        eprintln!(
            "[regalloc] fast {func_name}: {} values in slots, {} vregs assigned",
            frame.slot.len(),
            frame.assignment.len()
        );
    }

    Ok(GlobalRegAllocResult {
        per_block_insts: out,
        assignment: frame.assignment,
        callee_saved_used: used,
        unprecolored_params: Vec::new(),
        coalesce_aliases: BTreeMap::new(),
    })
}

fn is_flags(classes: &BTreeMap<VReg, RegClass>, v: VReg) -> bool {
    matches!(classes.get(&v), Some(RegClass::Flags))
}

fn class_of(classes: &BTreeMap<VReg, RegClass>, v: VReg) -> RegClass {
    classes.get(&v).copied().unwrap_or(RegClass::GPR)
}

fn claim_slot(frame: &mut Frame, slots: &mut SlotAllocator, v: VReg) {
    frame
        .slot
        .entry(v)
        .or_insert_with(|| slots.alloc(SlotOwner::Allocator));
}

/// One instruction becomes loads, itself, and a store.
fn expand(
    inst: &ScheduledInst,
    out: &mut Vec<ScheduledInst>,
    frame: &mut Frame,
    classes: &BTreeMap<VReg, RegClass>,
    gpr_pool: &[Reg],
    xmm_pool: &[Reg],
    func_name: &str,
) -> Result<(), String> {
    // A block parameter's marker defines nothing: the phi copies on the edges
    // have written its slot before the block runs.
    if matches!(inst.op, Op::Pure(PureOp::BlockParam(..))) {
        return Ok(());
    }

    // What this instruction has already committed, so a scratch does not land
    // on a register one of its arguments needs.
    let mut taken: BTreeSet<Reg> = BTreeSet::new();
    for &op in &inst.operands {
        if let Some(&r) = frame.precolored.get(&op) {
            taken.insert(r);
        }
    }
    if let Some(&r) = frame.precolored.get(&inst.dst) {
        taken.insert(r);
    }

    let mut operands: Vec<VReg> = Vec::with_capacity(inst.operands.len());
    for &op in &inst.operands {
        // Flags stay where the comparison left them. Nothing this pass inserts
        // between a comparison and its consumer writes EFLAGS: a spill load and
        // a spill store are both `mov`, which does not.
        if is_flags(classes, op) {
            operands.push(op);
            continue;
        }
        let Some(&slot) = frame.slot.get(&op) else {
            return Err(format!(
                "fast regalloc: {op:?} is read in '{func_name}' but is neither defined \
                 nor a block parameter, so there is no storage to read it from"
            ));
        };
        let class = class_of(classes, op);
        // A call argument must be in its argument register at the call, so the
        // load targets that register rather than a scratch.
        let reg = match frame.precolored.get(&op) {
            Some(&r) => r,
            None => pick(class, gpr_pool, xmm_pool, &taken, func_name)?,
        };
        taken.insert(reg);
        let tmp = frame.fresh();
        frame.assignment.insert(tmp, Assignment::Reg(reg));
        out.push(ScheduledInst {
            op: load_op(class, slot),
            dst: tmp,
            operands: vec![],
        });
        operands.push(tmp);
    }

    // Nothing to store: the op names no value, or its result is EFLAGS, which
    // no register holds. The op is asked rather than the class map, because the
    // map answers for a VReg and a comparison's dst can reach it as an ordinary
    // GPR from a block that saw it only as an operand -- storing that dst then
    // writes whatever the scratch register happened to hold.
    if inst.op.has_no_result() || inst.op.produces_flags() || is_flags(classes, inst.dst) {
        out.push(ScheduledInst {
            op: inst.op.clone(),
            dst: inst.dst,
            operands,
        });
        return Ok(());
    }

    let class = class_of(classes, inst.dst);
    let dst_reg = match frame.precolored.get(&inst.dst) {
        Some(&r) => r,
        None => pick(class, gpr_pool, xmm_pool, &taken, func_name)?,
    };
    let tmp = frame.fresh();
    frame.assignment.insert(tmp, Assignment::Reg(dst_reg));
    out.push(ScheduledInst {
        op: inst.op.clone(),
        dst: tmp,
        operands,
    });

    if let Some(&slot) = frame.slot.get(&inst.dst) {
        let sink = frame.fresh();
        out.push(ScheduledInst {
            op: store_op(class, slot),
            dst: sink,
            operands: vec![tmp],
        });
    }
    Ok(())
}

/// The first register of `class` this instruction has not already committed.
fn pick(
    class: RegClass,
    gpr_pool: &[Reg],
    xmm_pool: &[Reg],
    taken: &BTreeSet<Reg>,
    func_name: &str,
) -> Result<Reg, String> {
    let pool = match class {
        RegClass::GPR => gpr_pool,
        RegClass::XMM => xmm_pool,
        RegClass::Flags => return Err("fast regalloc: flags asked for a register".into()),
    };
    pool.iter()
        .copied()
        .find(|r| !taken.contains(r))
        .ok_or_else(|| {
            format!(
                "fast regalloc: one instruction in '{func_name}' needs more {class:?} \
                 registers than exist ({} in the pool). Nothing can relieve this: those \
                 values are live there because that instruction reads them",
                pool.len()
            )
        })
}

fn load_op(class: RegClass, slot: u32) -> Op {
    match class {
        RegClass::XMM => Op::Pseudo(PseudoOp::XmmSpillLoad(slot as i64)),
        _ => Op::Pseudo(PseudoOp::SpillLoad(slot as i64)),
    }
}

fn store_op(class: RegClass, slot: u32) -> Op {
    match class {
        RegClass::XMM => Op::Pseudo(PseudoOp::XmmSpillStore(slot as i64)),
        _ => Op::Pseudo(PseudoOp::SpillStore(slot as i64)),
    }
}
