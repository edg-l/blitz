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
//! whose block parameters outnumbered what was left -- `ROADMAP.md`'s closed
//! `-O0` allocator entry has what that ruled out.
//!
//! **What it does not do**: no interference graph, no liveness, no coalescing,
//! no splitting, no rematerialization. One pass over the instructions. It does
//! not need the pressure splitter in front of it either, and skipping that is
//! where the speed is: on a 6048-line input `-O0` goes 211.7ms to 150.5ms,
//! 1.41x. Nothing is live across an instruction, so the only shape this cannot
//! place is one instruction reading more operands than the machine has
//! registers -- which no amount of splitting would help.
//!
//! **What a slot cannot hold, this pass keeps out of one.** Three kinds of
//! value are not storage the model can name, and each is answered here rather
//! than left for something downstream to reconstruct:
//!
//! - **EFLAGS**, which no store reaches. A comparison's result is asked of the
//!   op rather than the class map, because the map answers for a VReg and a
//!   comparison's dst reaches it as an ordinary GPR from a block that saw it
//!   only as an operand.
//! - **A pair**, whose VReg holds neither of the halves it names. The
//!   projections are emitted next to the op that produces them and each gets a
//!   slot of its own -- see `plan_pairs`.
//! - **A jump's arguments**, which stay in their slots: the phi copies on the
//!   edge read them there, so a block with more parameters than the machine has
//!   registers costs no registers at all.
//!
//! A scratch also carries the type of the value it stands in for. Lowering
//! sizes an instruction from the type of the VReg it defines, and a scratch
//! with no entry in `vreg_types` makes it fall back to 64 bits.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::Type;
use crate::ir::op::{MachOp, Op, PseudoOp, PureOp};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::{ArgLoc, CALLEE_SAVED};
use crate::x86::reg::{Reg, RegClass};

use super::coloring::{allocatable_gpr_order, allocatable_xmm_order};
use super::interference::VRegSet;
use super::slots::{SlotAllocator, SlotOwner};
use super::{Assignment, GlobalRegAllocResult, build_vreg_classes_from_all_blocks};

/// Where every value of the function lives, and what the scratch VRegs minted
/// to move it resolve to.
struct Frame<'a> {
    /// One slot per value. Flags-class VRegs are absent: EFLAGS is in no
    /// register file and no store can reach it.
    slot: BTreeMap<VReg, u32>,
    /// The ABI register a value must occupy where it is read, from function
    /// parameters and call arguments. A load for one of these targets the
    /// register directly instead of a scratch.
    precolored: BTreeMap<VReg, Reg>,
    assignment: BTreeMap<VReg, Assignment>,
    /// The function's VReg types, extended as scratches are minted. Lowering
    /// sizes an instruction from the type of the VReg it defines and falls back
    /// to 64 bits when there is none, so a scratch standing in for a 32-bit
    /// value must state that value's type or `rol eax,7` is emitted as
    /// `rol rax,7`.
    types: &'a mut BTreeMap<VReg, Type>,
    next_vreg: u32,
}

impl Frame<'_> {
    /// A scratch standing in for `like`, carrying its type.
    fn fresh_like(&mut self, like: VReg) -> VReg {
        let v = VReg(self.next_vreg);
        self.next_vreg += 1;
        if let Some(ty) = self.types.get(&like).cloned() {
            self.types.insert(v, ty);
        }
        v
    }

    /// A scratch no instruction reads: the dead destination of a spill store.
    fn fresh_sink(&mut self) -> VReg {
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
    cfg_succs: &[Vec<usize>],
    block_param_vregs_per_block: &[VRegSet],
    func_name: &str,
    uses_frame_pointer: bool,
    arg_locs: &[ArgLoc],
    slots: &mut SlotAllocator,
    vreg_types: &mut BTreeMap<VReg, Type>,
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
        types: vreg_types,
        next_vreg,
    };

    // A pair VReg holds neither of the halves it names, so it gets one slot per
    // projected half rather than one of its own.
    let pairs = plan_pairs(block_schedules, &mut frame, slots, &classes);

    // Every value that is defined gets a slot, and so does every block
    // parameter -- a parameter is written by the phi copies on its edges rather
    // than by an instruction, so there is no definition in the block to find.
    // Flags are excluded: `RegClass::Flags` is EFLAGS, which no store reaches
    // and which the colouring allocator also leaves without a register.
    for insts in block_schedules {
        for inst in insts {
            if inst.op.has_no_result()
                || inst.op.produces_flags()
                || is_flags(&classes, inst.dst)
                || pairs.halves.contains_key(&inst.dst)
            {
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

    let ctx = Ctx {
        pairs: &pairs,
        classes: &classes,
        gpr_pool: allocatable_gpr_order(uses_frame_pointer),
        xmm_pool: allocatable_xmm_order(),
        arg_locs,
        func_name,
    };

    let mut out: Vec<Vec<ScheduledInst>> = Vec::with_capacity(block_schedules.len());
    for (block_idx, insts) in block_schedules.iter().enumerate() {
        // A block with no successors ends in `Ret`, and `Op::TerminatorArgs`
        // there names the returned value, which the return-register move reads
        // out of a register. Everywhere else it names the arguments of a jump or
        // a branch, which phi copies read out of their slots -- so those are
        // left in their slots, and a block with more parameters than the machine
        // has registers costs no registers at all.
        let returns = cfg_succs.get(block_idx).is_none_or(|s| s.is_empty());
        // The parameters of every block this one jumps to. An argument naming
        // one of them is the rotation a back edge can write -- see
        // `pass_through_terminator_args`.
        let mut successor_params = VRegSet::new();
        for &s in cfg_succs.get(block_idx).into_iter().flatten() {
            if let Some(params) = block_param_vregs_per_block.get(s) {
                successor_params.union_with(params);
            }
        }
        let mut block: Vec<ScheduledInst> = Vec::with_capacity(insts.len() * 3);
        // The parameters first, whatever order the schedule put their markers
        // in. A marker's position is not its value's position: every argument
        // register is caller-saved, so a call standing between the function's
        // entry and a marker takes that parameter with it. Storing them all
        // before anything else runs costs nothing, since a marker reads no
        // operand and so depends on nothing in the block.
        let is_param = |i: &ScheduledInst| matches!(i.op, Op::Pure(PureOp::Param(..)));
        for inst in insts.iter().filter(|i| is_param(i)) {
            expand(inst, &mut block, &mut frame, &ctx)?;
        }
        for inst in insts.iter().filter(|i| !is_param(i)) {
            if !returns && matches!(inst.op, Op::Pseudo(PseudoOp::TerminatorArgs(_))) {
                pass_through_terminator_args(
                    inst,
                    &mut block,
                    &mut frame,
                    &ctx,
                    &successor_params,
                    slots,
                )?;
                continue;
            }
            expand(inst, &mut block, &mut frame, &ctx)?;
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

/// Everything an expansion reads and none of it changes.
struct Ctx<'a> {
    /// The projections moved next to the ops that produce them.
    pairs: &'a Pairs,
    classes: &'a BTreeMap<VReg, RegClass>,
    gpr_pool: Vec<Reg>,
    xmm_pool: Vec<Reg>,
    /// Where the caller left each parameter.
    arg_locs: &'a [ArgLoc],
    /// Named by every error this pass can return.
    func_name: &'a str,
}

impl Ctx<'_> {
    fn class_of(&self, v: VReg) -> RegClass {
        class_of(self.classes, v)
    }

    fn is_flags(&self, v: VReg) -> bool {
        is_flags(self.classes, v)
    }

    /// The first register of `class` this instruction has not already committed.
    fn pick(&self, class: RegClass, taken: &BTreeSet<Reg>, op: &Op) -> Result<Reg, String> {
        let pool = match class {
            RegClass::GPR => &self.gpr_pool,
            RegClass::XMM => &self.xmm_pool,
            RegClass::Flags => return Err("fast regalloc: flags asked for a register".into()),
        };
        pool.iter()
            .copied()
            .find(|r| !taken.contains(r))
            .ok_or_else(|| {
                format!(
                    "fast regalloc: {op:?} in '{}' needs more {class:?} registers than exist \
                 ({} in the pool). Nothing can relieve this: those values are live there \
                 because that instruction reads them",
                    self.func_name,
                    pool.len(),
                )
            })
    }
}

fn is_flags(classes: &BTreeMap<VReg, RegClass>, v: VReg) -> bool {
    matches!(classes.get(&v), Some(RegClass::Flags))
}

fn class_of(classes: &BTreeMap<VReg, RegClass>, v: VReg) -> RegClass {
    classes.get(&v).copied().unwrap_or(RegClass::GPR)
}

fn is_division(op: &Op) -> bool {
    matches!(
        op,
        Op::Mach(MachOp::X86Idiv(_)) | Op::Mach(MachOp::X86Div(_))
    )
}

/// Which half of a pair a projection names, and where that half is kept.
struct Projected {
    half: PureOp,
    slot: u32,
}

/// The projections of every pair-producing op, moved next to the op itself.
struct Pairs {
    /// Pair VReg -> its projected halves, `Proj0` before `Proj1`.
    halves: BTreeMap<VReg, Vec<Projected>>,
    /// The projection VRegs those slots stand for, which therefore emit nothing.
    projections: BTreeSet<VReg>,
}

/// Give each projected half a slot and hand it to the op that produces it.
///
/// A pair VReg holds neither half of what it names. A division's quotient is in
/// RAX and its remainder in RDX; an ALU pair's value is in the register the op
/// wrote and its flags are in EFLAGS. A slot round trip between the op and its
/// projection loses all of that -- the register is reallocated by the next
/// instruction, and `lower.rs` reads the *adjacency* of a projection to decide
/// what the op means, so a subtraction whose difference is projected through a
/// slot is emitted as a flags-only `cmp` and the difference is never computed.
///
/// So the op writes each projected half into a slot where it is emitted, and the
/// projection becomes an alias for that slot: no instruction, and every later
/// read is a load of what the op stored.
///
/// A projection whose own result is EFLAGS is left alone: it names no storage,
/// its consumer reads the flags the op left, and nothing this pass inserts
/// between them writes them.
fn plan_pairs(
    block_schedules: &[Vec<ScheduledInst>],
    frame: &mut Frame<'_>,
    slots: &mut SlotAllocator,
    classes: &BTreeMap<VReg, RegClass>,
) -> Pairs {
    let produced: BTreeSet<VReg> = block_schedules
        .iter()
        .flatten()
        .filter(|i| !matches!(i.op, Op::Pure(PureOp::Proj0) | Op::Pure(PureOp::Proj1)))
        .map(|i| i.dst)
        .collect();

    let mut halves: BTreeMap<VReg, Vec<Projected>> = BTreeMap::new();
    let mut slot_of: BTreeMap<(VReg, u8), u32> = BTreeMap::new();
    let mut projections = BTreeSet::new();
    for inst in block_schedules.iter().flatten() {
        let half = match inst.op {
            Op::Pure(PureOp::Proj0) => PureOp::Proj0,
            Op::Pure(PureOp::Proj1) => PureOp::Proj1,
            _ => continue,
        };
        if is_flags(classes, inst.dst) {
            continue;
        }
        let Some(&pair) = inst.operands.first().filter(|v| produced.contains(v)) else {
            continue;
        };
        // One half can be projected in several blocks; they are one value in
        // one slot, and the op writes it once.
        let key = (pair, u8::from(half == PureOp::Proj1));
        let slot = match slot_of.entry(key) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let slot = *e.insert(slots.alloc(SlotOwner::Allocator));
                halves
                    .entry(pair)
                    .or_default()
                    .push(Projected { half, slot });
                slot
            }
        };
        frame.slot.insert(inst.dst, slot);
        projections.insert(inst.dst);
    }
    // `Proj0` before `Proj1`: taking a division's two results out of RAX and RDX
    // is one parallel copy, and `lower.rs` builds it from the order it finds.
    for list in halves.values_mut() {
        list.sort_by_key(|p| u8::from(p.half == PureOp::Proj1));
    }

    Pairs {
        halves,
        projections,
    }
}

fn claim_slot(frame: &mut Frame, slots: &mut SlotAllocator, v: VReg) {
    frame
        .slot
        .entry(v)
        .or_insert_with(|| slots.alloc(SlotOwner::Allocator));
}

/// A jump's or a branch's arguments, named where they already are.
///
/// The phi copies on the edge read each argument's slot and write the
/// parameter's, so nothing has to be in a register here. Saying `Slot` is what
/// tells `build_phi_copies` where to read: a VReg with no assignment at all
/// would be an argument the allocator placed nowhere, which is an error there.
///
/// One argument does not get to stay where it is: one that names a parameter of
/// the block being jumped to. Those copies are a parallel copy in memory, and a
/// back edge that rotates its parameters (`a, b = b, a`) makes one copy's
/// destination another's source, so emitting them in any order loses a value.
/// A copy into a slot of its own, before the terminator, takes the argument out
/// of the parameters' storage and leaves a list with no such pair in it.
fn pass_through_terminator_args(
    inst: &ScheduledInst,
    out: &mut Vec<ScheduledInst>,
    frame: &mut Frame<'_>,
    ctx: &Ctx<'_>,
    successor_params: &VRegSet,
    slots: &mut SlotAllocator,
) -> Result<(), String> {
    let mut args = Vec::with_capacity(inst.operands.len());
    for &op in &inst.operands {
        let Some(&slot) = frame.slot.get(&op) else {
            return Err(format!(
                "fast regalloc: terminator argument {op:?} in '{}' is neither \
                 defined nor a block parameter, so there is no storage to read it from",
                ctx.func_name
            ));
        };
        if !successor_params.contains(op.0 as usize) {
            frame.assignment.insert(op, Assignment::Slot(slot));
            args.push(op);
            continue;
        }
        let class = ctx.class_of(op);
        let reg = ctx.pick(class, &BTreeSet::new(), &inst.op)?;
        let held = frame.fresh_like(op);
        frame.assignment.insert(held, Assignment::Reg(reg));
        out.push(ScheduledInst {
            op: load_op(class, slot),
            dst: held,
            operands: vec![],
        });
        let shadow = frame.fresh_like(op);
        let shadow_slot = slots.alloc(SlotOwner::Allocator);
        frame
            .assignment
            .insert(shadow, Assignment::Slot(shadow_slot));
        let sink = frame.fresh_sink();
        out.push(ScheduledInst {
            op: store_op(class, shadow_slot),
            dst: sink,
            operands: vec![held],
        });
        args.push(shadow);
    }
    out.push(ScheduledInst {
        op: inst.op.clone(),
        dst: inst.dst,
        operands: args,
    });
    Ok(())
}

/// One instruction becomes loads, itself, and a store.
fn expand(
    inst: &ScheduledInst,
    out: &mut Vec<ScheduledInst>,
    frame: &mut Frame<'_>,
    ctx: &Ctx<'_>,
) -> Result<(), String> {
    // A block parameter's marker defines nothing: the phi copies on the edges
    // have written its slot before the block runs.
    if matches!(inst.op, Op::Pure(PureOp::BlockParam(..))) {
        return Ok(());
    }

    // A projection defines nothing either: its slot is the one its producer
    // stored that half into.
    if ctx.pairs.projections.contains(&inst.dst) {
        return Ok(());
    }

    // A register-passed parameter is already in its argument register before
    // the function's first instruction runs, so the store to its slot reads
    // that register and the parameter needs none of its own. Giving it a
    // scratch instead is what let every parameter of a six-argument function
    // share one: `pick` starts from an empty `taken` at each instruction, so
    // each marker in turn was handed the same first free register, and the six
    // stores then wrote one value into six slots. A stack-passed parameter is
    // not covered here -- lowering emits a load from the caller's frame at the
    // marker, which does need a register to land in.
    if let Op::Pure(PureOp::Param(param_idx, _)) = &inst.op
        && let Some(ArgLoc::Reg(abi_reg)) = ctx.arg_locs.get(*param_idx as usize)
    {
        let class = ctx.class_of(inst.dst);
        let tmp = frame.fresh_like(inst.dst);
        frame.assignment.insert(tmp, Assignment::Reg(*abi_reg));
        out.push(ScheduledInst {
            op: inst.op.clone(),
            dst: tmp,
            operands: vec![],
        });
        if let Some(&slot) = frame.slot.get(&inst.dst) {
            let sink = frame.fresh_sink();
            out.push(ScheduledInst {
                op: store_op(class, slot),
                dst: sink,
                operands: vec![tmp],
            });
        }
        return Ok(());
    }

    let division = is_division(&inst.op);
    // An ABI register is where a value has to be *at a call*, not a property it
    // carries everywhere. A call's own barrier honours it; any other instruction
    // reads the value out of its slot into whatever register is free, because
    // two of its operands can carry the same colour -- a call result and some
    // other call's first floating-point argument are both XMM0, and adding them
    // loaded both into it and added the second to itself.
    let calls = matches!(
        inst.op,
        Op::Pseudo(PseudoOp::CallResult(..) | PseudoOp::VoidCallBarrier)
    );

    // What this instruction has already committed, so a scratch does not land
    // on a register one of its arguments needs.
    let mut taken: BTreeSet<Reg> = BTreeSet::new();
    if calls {
        for &op in &inst.operands {
            if let Some(&r) = frame.precolored.get(&op) {
                taken.insert(r);
            }
        }
    }
    if let Some(&r) = frame.precolored.get(&inst.dst) {
        taken.insert(r);
    }
    if division {
        taken.insert(Reg::RAX);
        taken.insert(Reg::RDX);
    }

    // The registers earlier operands of this instruction have committed to.
    // `taken` cannot answer that: it is seeded with every operand's colour up
    // front, so that a scratch does not land on one an operand still needs.
    let mut committed: BTreeSet<Reg> = BTreeSet::new();
    let mut operands: Vec<VReg> = Vec::with_capacity(inst.operands.len());
    for (idx, &op) in inst.operands.iter().enumerate() {
        // Flags stay where the comparison left them. Nothing this pass inserts
        // between a comparison and its consumer writes EFLAGS: a spill load and
        // a spill store are both `mov`, which does not.
        if ctx.is_flags(op) {
            operands.push(op);
            continue;
        }
        let Some(&slot) = frame.slot.get(&op) else {
            return Err(format!(
                "fast regalloc: {op:?} is read in '{}' but is neither defined \
                 nor a block parameter, so there is no storage to read it from",
                ctx.func_name
            ));
        };
        let class = ctx.class_of(op);
        // `idiv` reads its dividend in RDX:RAX and writes both, so the
        // dividend's load targets RAX and the divisor's may target neither. A
        // call argument must be in its argument register where the call reads
        // it, so its load targets that rather than a scratch.
        // A call argument's colour is a hint, not a constraint: `setup_call_args`
        // moves each argument from wherever its operand landed into the ABI
        // register for its position, so the only thing that has to hold here is
        // that no two operands share a register. Two can carry the same colour --
        // the same value passed twice, or a value that is also another call's
        // argument at a different position, since the map is keyed by value and
        // the last writer wins.
        let colour = frame
            .precolored
            .get(&op)
            .copied()
            .filter(|r| calls && !committed.contains(r));
        let reg = match colour {
            _ if division && idx == 0 => Reg::RAX,
            Some(r) => r,
            None => ctx.pick(class, &taken, &inst.op)?,
        };
        committed.insert(reg);
        taken.insert(reg);
        let tmp = frame.fresh_like(op);
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
    if inst.op.has_no_result() || inst.op.produces_flags() || ctx.is_flags(inst.dst) {
        out.push(ScheduledInst {
            op: inst.op.clone(),
            dst: inst.dst,
            operands,
        });
        return Ok(());
    }

    // A pair-producing op writes its halves where its projections can still see
    // them: in the registers it just wrote, with nothing in between. Each
    // projection is emitted here and its slot filled from the register the
    // projection copies out of.
    if let Some(halves) = ctx.pairs.halves.get(&inst.dst) {
        let pair_reg = if division {
            Reg::RAX
        } else {
            ctx.pick(ctx.class_of(inst.dst), &taken, &inst.op)?
        };
        taken.insert(pair_reg);
        let pair = frame.fresh_like(inst.dst);
        frame.assignment.insert(pair, Assignment::Reg(pair_reg));
        out.push(ScheduledInst {
            op: inst.op.clone(),
            dst: pair,
            operands,
        });
        // The projections follow it directly: lowering takes a division's two
        // results out of RAX and RDX as one parallel copy, asserts on anything
        // standing between the division and them, and resolves that copy
        // through R11. Distinct registers, none of them R11, for the same
        // reason.
        taken.insert(Reg::R11);
        let mut copied = Vec::with_capacity(halves.len());
        for projected in halves {
            let class = ctx.class_of(inst.dst);
            let reg = ctx.pick(class, &taken, &inst.op)?;
            taken.insert(reg);
            let half = frame.fresh_like(inst.dst);
            frame.assignment.insert(half, Assignment::Reg(reg));
            out.push(ScheduledInst {
                op: Op::Pure(projected.half.clone()),
                dst: half,
                operands: vec![pair],
            });
            copied.push((half, class, projected.slot));
        }
        for (half, class, slot) in copied {
            let sink = frame.fresh_sink();
            out.push(ScheduledInst {
                op: store_op(class, slot),
                dst: sink,
                operands: vec![half],
            });
        }
        return Ok(());
    }

    let class = ctx.class_of(inst.dst);
    let dst_reg = match frame.precolored.get(&inst.dst) {
        Some(&r) => r,
        None => ctx.pick(class, &taken, &inst.op)?,
    };
    let tmp = frame.fresh_like(inst.dst);
    frame.assignment.insert(tmp, Assignment::Reg(dst_reg));
    out.push(ScheduledInst {
        op: inst.op.clone(),
        dst: tmp,
        operands,
    });

    if let Some(&slot) = frame.slot.get(&inst.dst) {
        let sink = frame.fresh_sink();
        out.push(ScheduledInst {
            op: store_op(class, slot),
            dst: sink,
            operands: vec![tmp],
        });
    }
    Ok(())
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
