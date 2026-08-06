use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::{Op, PseudoOp};
use crate::schedule::scheduler::ScheduledInst;

use super::slots::{SlotAllocator, SlotOwner};

pub(crate) const LOOP_DEPTH_PENALTY_BASE: u64 = 10;

// ── Spill/reload pseudo-op markers ───────────────────────────────────────────
//
// Spills and reloads are encoded as dedicated Op variants:
//   Op::Pseudo(PseudoOp::SpillStore(slot))    - GPR spill store; operands = [vreg_being_spilled]
//   Op::Pseudo(PseudoOp::SpillLoad(slot))     - GPR spill load;  dst = reload VReg
//   Op::Pseudo(PseudoOp::XmmSpillStore(slot)) - XMM spill store; operands = [vreg_being_spilled]
//   Op::Pseudo(PseudoOp::XmmSpillLoad(slot))  - XMM spill load;  dst = reload VReg
//
// These are lowered to real MovMR/MovRM (GPR) or MovsdMR/MovsdRM (XMM) by the backend.

pub fn is_spill_store(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Pseudo(PseudoOp::SpillStore(_)))
}

pub fn is_spill_load(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Pseudo(PseudoOp::SpillLoad(_)))
}

pub fn is_xmm_spill_store(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Pseudo(PseudoOp::XmmSpillStore(_)))
}

pub fn is_xmm_spill_load(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Pseudo(PseudoOp::XmmSpillLoad(_)))
}

pub fn spill_slot_of(inst: &ScheduledInst) -> u32 {
    match &inst.op {
        Op::Pseudo(PseudoOp::SpillStore(slot))
        | Op::Pseudo(PseudoOp::SpillLoad(slot))
        | Op::Pseudo(PseudoOp::XmmSpillStore(slot))
        | Op::Pseudo(PseudoOp::XmmSpillLoad(slot)) => *slot as u32,
        _ => unreachable!("spill_slot_of called on non-spill inst"),
    }
}

// ── Spill selection (10.9) ────────────────────────────────────────────────────

/// Compute the live range length (definition to last use) for each VReg.
///
/// For each VReg index, scans instructions to find the defining position
/// (where `inst.dst == VReg(idx)`) and the last use position (last appearance
/// in any `inst.operands`). Range length = last_use - def_pos (or 1 if not found).
pub fn compute_live_range_length(insts: &[ScheduledInst]) -> BTreeMap<usize, usize> {
    let mut def_pos: BTreeMap<usize, usize> = BTreeMap::new();
    let mut last_use: BTreeMap<usize, usize> = BTreeMap::new();

    for (i, inst) in insts.iter().enumerate() {
        let dst_idx = inst.dst.0 as usize;
        def_pos.entry(dst_idx).or_insert(i);

        for &op in &inst.operands {
            let op_idx = op.0 as usize;
            last_use.insert(op_idx, i);
        }
    }

    let mut range_lengths = BTreeMap::new();
    let all_vregs: BTreeSet<usize> = def_pos.keys().chain(last_use.keys()).copied().collect();
    for idx in all_vregs {
        let dp = def_pos.get(&idx).copied().unwrap_or(0);
        let lu = last_use.get(&idx).copied().unwrap_or(dp);
        let len = if lu >= dp { lu - dp } else { 1 };
        range_lengths.insert(idx, len.max(1));
    }

    range_lengths
}

// ── Rematerialization check ────────────────────────────────────────────────────

/// Returns true if the VReg defined by `inst` can be rematerialized
/// (i.e., cheaply recomputed instead of spilled to memory).
pub fn is_rematerializable(inst: &ScheduledInst) -> bool {
    inst.op.is_rematerializable()
}

/// Collect the set of VRegs that are call arguments (operands of CallResult or
/// VoidCallBarrier instructions). These VRegs must NOT be rematerialized away
/// from their use position, because the call needs the value in a register at
/// the call point and rematerialization would shorten the live range past
/// call clobber points.
pub fn collect_call_arg_vregs(insts: &[ScheduledInst]) -> BTreeSet<usize> {
    let mut call_args = BTreeSet::new();
    for inst in insts {
        if matches!(
            inst.op,
            Op::Pseudo(PseudoOp::CallResult(_, _)) | Op::Pseudo(PseudoOp::VoidCallBarrier)
        ) {
            for &op in &inst.operands {
                call_args.insert(op.0 as usize);
            }
        }
    }
    call_args
}

// ── Spill code insertion (10.10) ──────────────────────────────────────────────

/// Insert spill/reload code for the given set of spilled VReg indices.
///
/// For each spilled VReg:
/// - If rematerializable (Iconst): re-emit the defining instruction before
///   each use as a new short-lived VReg (10.12 rematerialization).
/// - Otherwise: insert a SpillStore after the def, and a SpillLoad before
///   each use as a new short-lived VReg.
///
/// Returns a mapping from each original spilled VReg to the set of new
/// reload VRegs that replace its uses (one per use site).
///
/// `next_vreg` is updated to allocate new VReg indices.
pub fn insert_spills(
    insts: &mut Vec<ScheduledInst>,
    spilled: &BTreeSet<usize>,
    slots: &mut SlotAllocator,
    next_vreg: &mut u32,
    vreg_classes: &BTreeMap<VReg, crate::x86::reg::RegClass>,
) -> BTreeMap<VReg, Vec<VReg>> {
    if spilled.is_empty() {
        return BTreeMap::new();
    }
    let plan = SpillPlacement::plan(std::slice::from_ref(&*insts), spilled, slots);
    let mut reload_map = BTreeMap::new();
    plan.apply_to(insts, next_vreg, vreg_classes, &mut reload_map);
    reload_map
}

/// Spill every VReg in `spilled` across every block of a function.
///
/// The function-wide form of [`insert_spills`], and the difference is the whole
/// point: a slot is allocated **once per spilled VReg**, not once per block, and
/// the defining instruction a reload rematerializes from may live in a different
/// block than the use. Spilling per block with per-block slots would give one
/// value two cells and store it into the one nobody reads.
///
/// Spill-everywhere: a store after each def, a reload before each use, each
/// reload with a VReg of its own. What survives across a block boundary is the
/// slot, so the value stops being live-out and the next round's liveness sees
/// that without being told.
pub fn insert_spills_global(
    block_schedules: &mut [Vec<ScheduledInst>],
    spilled: &BTreeSet<usize>,
    slots: &mut SlotAllocator,
    next_vreg: &mut u32,
    vreg_classes: &BTreeMap<VReg, crate::x86::reg::RegClass>,
) {
    if spilled.is_empty() {
        return;
    }
    let plan = SpillPlacement::plan(block_schedules, spilled, slots);
    let mut reload_map = BTreeMap::new();
    for insts in block_schedules.iter_mut() {
        plan.apply_to(insts, next_vreg, vreg_classes, &mut reload_map);
    }
}

/// What spilling a set of VRegs needs decided once, before any list is rewritten:
/// where each one is defined, which are call arguments, and which slot each takes.
struct SpillPlacement<'a> {
    spilled: &'a BTreeSet<usize>,
    def_ops: BTreeMap<usize, ScheduledInst>,
    call_arg_vregs: BTreeSet<usize>,
    vreg_to_slot: BTreeMap<usize, u32>,
}

impl<'a> SpillPlacement<'a> {
    fn plan(
        block_schedules: &[Vec<ScheduledInst>],
        spilled: &'a BTreeSet<usize>,
        slots: &mut SlotAllocator,
    ) -> Self {
        // VReg -> defining instruction, for rematerialization.
        let def_ops: BTreeMap<usize, ScheduledInst> = block_schedules
            .iter()
            .flatten()
            .filter(|inst| spilled.contains(&(inst.dst.0 as usize)))
            .map(|inst| {
                (
                    inst.dst.0 as usize,
                    ScheduledInst {
                        op: inst.op.clone(),
                        dst: inst.dst,
                        operands: inst.operands.clone(),
                    },
                )
            })
            .collect();

        // Call-arg VRegs must NOT be rematerialized even where the defining op
        // is an Iconst or a StackAddr: the value has to be live at the call
        // point, and rematerializing shortens its range past the clobber.
        let mut call_arg_vregs: BTreeSet<usize> = BTreeSet::new();
        for insts in block_schedules {
            call_arg_vregs.extend(collect_call_arg_vregs(insts));
        }

        // One slot per spilled VReg that needs one, for the whole function.
        let mut vreg_to_slot: BTreeMap<usize, u32> = BTreeMap::new();
        for &idx in spilled {
            let is_call_arg = call_arg_vregs.contains(&idx);
            let needs_slot = if let Some(def) = def_ops.get(&idx) {
                !is_rematerializable(def) || is_call_arg
            } else {
                false
            };
            if needs_slot {
                let slot = slots.alloc(SlotOwner::Allocator);
                vreg_to_slot.insert(idx, slot);
            }
        }

        SpillPlacement {
            spilled,
            def_ops,
            call_arg_vregs,
            vreg_to_slot,
        }
    }
}

impl SpillPlacement<'_> {
    /// Rewrite one instruction list: a reload before each use, a store after
    /// each def.
    fn apply_to(
        &self,
        insts: &mut Vec<ScheduledInst>,
        next_vreg: &mut u32,
        vreg_classes: &BTreeMap<VReg, crate::x86::reg::RegClass>,
        reload_map: &mut BTreeMap<VReg, Vec<VReg>>,
    ) {
        let spilled = self.spilled;
        let def_ops = &self.def_ops;
        let call_arg_vregs = &self.call_arg_vregs;
        let vreg_to_slot = &self.vreg_to_slot;

        // We need to process the instruction list and insert spill/reload code.
        // We do a single pass, building a new instruction list.
        let old_insts = std::mem::take(insts);
        let mut new_insts: Vec<ScheduledInst> = Vec::with_capacity(old_insts.len() * 2);

        for mut inst in old_insts {
            // Before this instruction, insert reloads for any spilled operands.
            //
            // One per (instruction, spilled VReg), not one per occurrence: every
            // operand of an instruction is read at the same point, so a value
            // named twice needs one register holding it, not two. A fresh VReg
            // per occurrence puts k copies of one value live at one program
            // point, and a terminator's argument list names the same value as
            // several block arguments routinely.
            let mut reloaded_here: BTreeMap<usize, VReg> = BTreeMap::new();
            let mut new_operands = Vec::with_capacity(inst.operands.len());
            for &op in &inst.operands {
                let op_idx = op.0 as usize;
                if spilled.contains(&op_idx) {
                    if let Some(&already) = reloaded_here.get(&op_idx) {
                        new_operands.push(already);
                        continue;
                    }
                    // Replace with a reload VReg. Call-arg VRegs must NOT use
                    // rematerialization: the original def must stay alive at the
                    // call point with proper interference against call clobbers.
                    let reload_vreg = if let Some(def) = def_ops.get(&op_idx)
                        && is_rematerializable(def)
                        && !call_arg_vregs.contains(&op_idx)
                    {
                        // Rematerialization: re-emit the defining instruction.
                        let new_vreg = VReg(*next_vreg);
                        *next_vreg += 1;
                        let remat_inst = ScheduledInst {
                            op: def.op.clone(),
                            dst: new_vreg,
                            operands: def.operands.clone(),
                        };
                        new_insts.push(remat_inst);
                        new_vreg
                    } else if let Some(&slot) = vreg_to_slot.get(&op_idx) {
                        // Check if we already inserted a reload for this use.
                        // Each use gets its own reload VReg (short-lived).
                        let new_vreg = VReg(*next_vreg);
                        *next_vreg += 1;
                        let is_xmm = super::is_xmm_vreg(op, vreg_classes);
                        let load_op = if is_xmm {
                            Op::Pseudo(PseudoOp::XmmSpillLoad(slot as i64))
                        } else {
                            Op::Pseudo(PseudoOp::SpillLoad(slot as i64))
                        };
                        let load_inst = ScheduledInst {
                            op: load_op,
                            dst: new_vreg,
                            operands: vec![],
                        };
                        new_insts.push(load_inst);
                        reload_map.entry(op).or_default().push(new_vreg);
                        new_vreg
                    } else {
                        // Should not happen: spilled but no slot and not remat.
                        op
                    };
                    reloaded_here.insert(op_idx, reload_vreg);
                    new_operands.push(reload_vreg);
                } else {
                    new_operands.push(op);
                }
            }
            inst.operands = new_operands;

            let dst_idx = inst.dst.0 as usize;
            let is_spill_def = spilled.contains(&dst_idx);

            // For rematerializable spilled VRegs (that are NOT call args), drop
            // the original definition — uses are replaced by fresh remat copies.
            // Call-arg VRegs keep their def so they remain live at call points.
            if is_spill_def
                && def_ops.get(&dst_idx).is_some_and(is_rematerializable)
                && !call_arg_vregs.contains(&dst_idx)
            {
                // Skip the original definition.
            } else {
                new_insts.push(inst);
            }

            // After the def of a spilled VReg, insert a SpillStore (if not remat).
            if is_spill_def && let Some(&slot) = vreg_to_slot.get(&dst_idx) {
                let spilled_vreg = VReg(dst_idx as u32);
                let is_xmm = super::is_xmm_vreg(spilled_vreg, vreg_classes);
                let store_op = if is_xmm {
                    Op::Pseudo(PseudoOp::XmmSpillStore(slot as i64))
                } else {
                    Op::Pseudo(PseudoOp::SpillStore(slot as i64))
                };
                let dummy_dst = VReg(*next_vreg);
                *next_vreg += 1;
                let store_inst = ScheduledInst {
                    op: store_op,
                    dst: dummy_dst,
                    operands: vec![spilled_vreg],
                };
                new_insts.push(store_inst);
            }
        }

        *insts = new_insts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::{Op, PureOp};
    use crate::ir::types::Type;

    fn iconst_inst(dst: u32, val: i64) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::Iconst(val, Type::I64)),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::Proj0),
            dst: VReg(dst),
            operands: vec![VReg(src)],
        }
    }

    // SpillStore is inserted after def, SpillLoad before each use.
    #[test]
    fn spill_store_and_load_inserted() {
        // v0 = iconst (non-trivial, we need a non-Iconst def for spill store)
        // simulate by using a two-operand inst as def.
        // Actually let's use: v0 = proj0(v_dummy), v1 = use(v0), v2 = use(v0)
        let insts_base = vec![
            ScheduledInst {
                op: Op::Pure(PureOp::Proj0),
                dst: VReg(0),
                operands: vec![VReg(99)], // dummy operand
            },
            use_inst(1, 0),
            use_inst(2, 0),
        ];

        let mut insts = insts_base.clone();
        let mut spilled = BTreeSet::new();
        spilled.insert(0usize); // spill v0
        let mut slots = SlotAllocator::new();
        let mut next_vreg = 100u32;

        insert_spills(
            &mut insts,
            &spilled,
            &mut slots,
            &mut next_vreg,
            &BTreeMap::new(),
        );

        assert_eq!(slots.count(), 1);

        // There should be a SpillStore after inst 0.
        let store_pos = insts
            .iter()
            .position(is_spill_store)
            .expect("SpillStore must be present");
        let def_pos = insts
            .iter()
            .position(|i| i.dst == VReg(0))
            .expect("original def must be present");
        assert!(
            store_pos > def_pos,
            "SpillStore must come after def: store={store_pos} def={def_pos}"
        );

        // There should be two SpillLoads (one per use).
        let load_count = insts.iter().filter(|i| is_spill_load(i)).count();
        assert_eq!(load_count, 2, "two SpillLoads expected (one per use)");
    }

    // loop-depth penalty — VReg inside a loop is preferred to NOT be spilled.
    //
    // Given two candidates with equal next-use distance but different loop depths,
    // the one outside the loop (depth=0) should be spilled before the loop-body one.

    // Rematerialization: Iconst is re-emitted before each use, no SpillStore.
    #[test]
    fn rematerialization_no_store() {
        let mut insts = vec![
            iconst_inst(0, 42), // v0 = iconst(42)
            use_inst(1, 0),     // v1 = use(v0)
            use_inst(2, 0),     // v2 = use(v0)
        ];
        let mut spilled = BTreeSet::new();
        spilled.insert(0usize);
        let mut slots = SlotAllocator::new();
        let mut next_vreg = 10u32;

        insert_spills(
            &mut insts,
            &spilled,
            &mut slots,
            &mut next_vreg,
            &BTreeMap::new(),
        );

        // No SpillStore: constants are rematerializable.
        assert!(
            !insts.iter().any(is_spill_store),
            "no SpillStore for rematerializable Iconst"
        );
        // No SpillLoad either: replaced by re-emitted Iconst.
        assert!(
            !insts.iter().any(is_spill_load),
            "no SpillLoad for rematerializable Iconst"
        );
        assert_eq!(slots.count(), 0);
    }

    // XMM VReg spill inserts XMM-specific markers (not GPR markers).
    //
    // When a VReg is classified as XMM (via vreg_classes), insert_spills must
    // emit XMM_SPILL_STORE_TYPE / XMM_SPILL_LOAD_TYPE sentinels instead of the
    // GPR sentinels. The compile.rs lowering then emits MOVSD instead of MOV.
    #[test]
    fn xmm_spill_uses_xmm_markers() {
        use crate::x86::reg::RegClass;

        // Simulate an XMM VReg: v0 = Proj0 (non-remat, will be spilled).
        // v1 = use(v0); v2 = use(v0)
        let mut insts = vec![
            ScheduledInst {
                op: Op::Pure(PureOp::Proj0),
                dst: VReg(0),
                operands: vec![VReg(99)],
            },
            use_inst(1, 0),
            use_inst(2, 0),
        ];

        let mut spilled = BTreeSet::new();
        spilled.insert(0usize);
        let mut slots = SlotAllocator::new();
        let mut next_vreg = 100u32;

        // Mark v0 as XMM class.
        let mut vreg_classes = BTreeMap::new();
        vreg_classes.insert(VReg(0), RegClass::XMM);

        insert_spills(
            &mut insts,
            &spilled,
            &mut slots,
            &mut next_vreg,
            &vreg_classes,
        );

        // There should be an XMM SpillStore (not a GPR SpillStore).
        assert!(
            insts.iter().any(is_xmm_spill_store),
            "XMM VReg spill must produce XMM_SPILL_STORE_TYPE marker"
        );
        assert!(
            !insts.iter().any(is_spill_store),
            "XMM VReg spill must NOT produce GPR SPILL_STORE_TYPE marker"
        );

        // There should be XMM SpillLoads before each use.
        let xmm_load_count = insts.iter().filter(|i| is_xmm_spill_load(i)).count();
        assert_eq!(
            xmm_load_count, 2,
            "two XMM SpillLoads expected (one per use)"
        );
        assert!(
            !insts.iter().any(is_spill_load),
            "XMM VReg spill must NOT produce GPR SPILL_LOAD_TYPE marker"
        );
    }

    // Two candidates with same next-use but different degree*range. The one with
    // higher degree*range_length should be selected (more pressure relief).

    // Best candidate by score is in excluded set, verify next-best is chosen.
}
