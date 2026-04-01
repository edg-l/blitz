use std::collections::{HashMap, HashSet};

use crate::egraph::extract::VReg;
use crate::ir::op::Op;
use crate::schedule::scheduler::ScheduledInst;

use super::interference::InterferenceGraph;
use super::liveness::LivenessInfo;

// ── Spill/reload pseudo-op markers ───────────────────────────────────────────
//
// Spills and reloads are encoded as dedicated Op variants:
//   Op::SpillStore(slot)    - GPR spill store; operands = [vreg_being_spilled]
//   Op::SpillLoad(slot)     - GPR spill load;  dst = reload VReg
//   Op::XmmSpillStore(slot) - XMM spill store; operands = [vreg_being_spilled]
//   Op::XmmSpillLoad(slot)  - XMM spill load;  dst = reload VReg
//
// These are lowered to real MovMR/MovRM (GPR) or MovsdMR/MovsdRM (XMM) by the backend.

pub fn is_spill_store(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::SpillStore(_))
}

pub fn is_spill_load(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::SpillLoad(_))
}

pub fn is_xmm_spill_store(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::XmmSpillStore(_))
}

pub fn is_xmm_spill_load(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::XmmSpillLoad(_))
}

pub fn spill_slot_of(inst: &ScheduledInst) -> u32 {
    match &inst.op {
        Op::SpillStore(slot)
        | Op::SpillLoad(slot)
        | Op::XmmSpillStore(slot)
        | Op::XmmSpillLoad(slot) => *slot as u32,
        _ => unreachable!("spill_slot_of called on non-spill inst"),
    }
}

// ── Spill selection (10.9) ────────────────────────────────────────────────────

/// Select a VReg to spill using a farthest-next-use heuristic with loop-depth penalty.
///
/// Among all VRegs that are live (present in the interference graph) and not
/// rematerializable, picks the one whose next use is farthest from the point
/// where register pressure exceeds `available_regs`. VRegs defined inside a
/// loop (higher `loop_depths`) are penalized so they are less likely to be
/// spilled.
///
/// Returns `None` if no spill candidate is found (shouldn't happen if the
/// chromatic number truly exceeds available_regs).
pub fn select_spill(
    graph: &InterferenceGraph,
    liveness: &LivenessInfo,
    insts: &[ScheduledInst],
    available_regs: u32,
    loop_depths: &HashMap<VReg, u32>,
) -> Option<usize> {
    // Find the instruction index where we first exceed register pressure.
    if let Some(pressure_point) = find_pressure_point(liveness, available_regs) {
        let next_use = compute_next_use(insts, pressure_point);
        let live_at_pressure = &liveness.live_at[pressure_point];

        // Consider all VRegs live at the pressure point (including remat).
        let candidates: Vec<usize> = live_at_pressure
            .iter()
            .map(|v| v.0 as usize)
            .filter(|&idx| idx < graph.num_vregs)
            .collect();

        // Pick the candidate with the farthest next use, penalized by loop depth.
        if let Some(best) = candidates.into_iter().max_by_key(|&idx| {
            let next = next_use.get(&idx).copied().unwrap_or(usize::MAX);
            let depth = loop_depths.get(&VReg(idx as u32)).copied().unwrap_or(0);
            let penalty = 10u64.saturating_pow(depth);
            (next as u64).saturating_div(penalty)
        }) {
            return Some(best);
        }
    }

    // Fallback: coloring overestimates but no point exceeds available_regs.
    // Pick the VReg with the highest interference degree.
    (0..graph.num_vregs)
        .filter(|&idx| !graph.adj[idx].is_empty())
        .max_by_key(|&idx| {
            let degree = graph.adj[idx].len();
            let depth = loop_depths.get(&VReg(idx as u32)).copied().unwrap_or(0);
            let penalty = 10usize.saturating_pow(depth);
            degree / penalty.max(1)
        })
}

fn find_pressure_point(liveness: &LivenessInfo, available_regs: u32) -> Option<usize> {
    for (i, live_set) in liveness.live_at.iter().enumerate() {
        // live_at[i] excludes the destination of instruction i, but the dst
        // also needs a register. True pressure = live_at[i].len() + 1.
        if live_set.len() >= available_regs as usize {
            return Some(i);
        }
    }
    None
}

fn compute_next_use(insts: &[ScheduledInst], from: usize) -> HashMap<usize, usize> {
    let mut next_use: HashMap<usize, usize> = HashMap::new();
    for (i, inst) in insts.iter().enumerate().skip(from) {
        for &op in &inst.operands {
            let idx = op.0 as usize;
            next_use.entry(idx).or_insert(i);
        }
    }
    next_use
}

// ── Rematerialization check ────────────────────────────────────────────────────

/// Returns true if the VReg defined by `inst` can be rematerialized
/// (i.e., cheaply recomputed instead of spilled to memory).
/// Both Iconst and StackAddr have no dependencies and produce constants.
pub fn is_rematerializable(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Iconst(_, _) | Op::StackAddr(_))
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
    spilled: &HashSet<usize>,
    spill_slots: &mut u32,
    next_vreg: &mut u32,
    vreg_classes: &HashMap<VReg, crate::x86::reg::RegClass>,
) -> HashMap<VReg, Vec<VReg>> {
    if spilled.is_empty() {
        return HashMap::new();
    }

    // Build a map of VReg -> defining instruction op (for rematerialization).
    let def_ops: HashMap<usize, ScheduledInst> = insts
        .iter()
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

    // Assign spill slots to non-rematerializable VRegs.
    let mut vreg_to_slot: HashMap<usize, u32> = HashMap::new();
    for &idx in spilled {
        if let Some(def) = def_ops.get(&idx)
            && !is_rematerializable(def)
        {
            let slot = *spill_slots;
            *spill_slots += 1;
            vreg_to_slot.insert(idx, slot);
        }
    }

    let mut reload_map: HashMap<VReg, Vec<VReg>> = HashMap::new();

    // We need to process the instruction list and insert spill/reload code.
    // We do a single pass, building a new instruction list.
    let old_insts = std::mem::take(insts);
    let mut new_insts: Vec<ScheduledInst> = Vec::with_capacity(old_insts.len() * 2);

    // Track current reload VRegs for each spilled VReg.
    // Maps original VReg index -> current reload VReg (if a reload was just inserted).
    let mut current_reload: HashMap<usize, VReg> = HashMap::new();

    for mut inst in old_insts {
        // Before this instruction, insert reloads for any spilled operands.
        let mut new_operands = Vec::with_capacity(inst.operands.len());
        for &op in &inst.operands {
            let op_idx = op.0 as usize;
            if spilled.contains(&op_idx) {
                // Replace with a reload VReg.
                let reload_vreg = if let Some(def) = def_ops.get(&op_idx)
                    && is_rematerializable(def)
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
                    let is_xmm = vreg_classes
                        .get(&op)
                        .copied()
                        .map(|c| c == crate::x86::reg::RegClass::XMM)
                        .unwrap_or(false);
                    let load_op = if is_xmm {
                        Op::XmmSpillLoad(slot as i64)
                    } else {
                        Op::SpillLoad(slot as i64)
                    };
                    let load_inst = ScheduledInst {
                        op: load_op,
                        dst: new_vreg,
                        operands: vec![],
                    };
                    new_insts.push(load_inst);
                    reload_map.entry(op).or_default().push(new_vreg);
                    current_reload.insert(op_idx, new_vreg);
                    new_vreg
                } else {
                    // Should not happen: spilled but no slot and not remat.
                    op
                };
                new_operands.push(reload_vreg);
            } else {
                new_operands.push(op);
            }
        }
        inst.operands = new_operands;

        let dst_idx = inst.dst.0 as usize;
        let is_spill_def = spilled.contains(&dst_idx);

        // For rematerializable spilled VRegs, drop the original definition
        // entirely — uses are replaced by fresh remat copies above. Keeping
        // the dead def would preserve its live range via block_live_out.
        if is_spill_def
            && def_ops
                .get(&dst_idx)
                .map_or(false, |d| is_rematerializable(d))
        {
            // Skip the original definition.
        } else {
            new_insts.push(inst);
        }

        // After the def of a spilled VReg, insert a SpillStore (if not remat).
        if is_spill_def && let Some(&slot) = vreg_to_slot.get(&dst_idx) {
            let spilled_vreg = VReg(dst_idx as u32);
            let is_xmm = vreg_classes
                .get(&spilled_vreg)
                .copied()
                .map(|c| c == crate::x86::reg::RegClass::XMM)
                .unwrap_or(false);
            let store_op = if is_xmm {
                Op::XmmSpillStore(slot as i64)
            } else {
                Op::SpillStore(slot as i64)
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

    // Build reload_map entries for rematerialized VRegs (already handled inline above).
    reload_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn iconst_inst(dst: u32, val: i64) -> ScheduledInst {
        ScheduledInst {
            op: Op::Iconst(val, Type::I64),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Proj0,
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
                op: Op::Proj0,
                dst: VReg(0),
                operands: vec![VReg(99)], // dummy operand
            },
            use_inst(1, 0),
            use_inst(2, 0),
        ];

        let mut insts = insts_base.clone();
        let mut spilled = HashSet::new();
        spilled.insert(0usize); // spill v0
        let mut spill_slots = 0u32;
        let mut next_vreg = 100u32;

        insert_spills(
            &mut insts,
            &spilled,
            &mut spill_slots,
            &mut next_vreg,
            &HashMap::new(),
        );

        // spill_slots should now be 1.
        assert_eq!(spill_slots, 1);

        // There should be a SpillStore after inst 0.
        let store_pos = insts
            .iter()
            .position(|i| is_spill_store(i))
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

    // Phase 4.2: loop-depth penalty — VReg inside a loop is preferred to NOT be spilled.
    //
    // Given two candidates with equal next-use distance but different loop depths,
    // the one outside the loop (depth=0) should be spilled before the loop-body one.
    #[test]
    fn loop_depth_penalty_prefers_outer_spill() {
        use super::super::interference::InterferenceGraph;
        use super::super::liveness::LivenessInfo;
        use crate::x86::reg::RegClass;

        // Construct a minimal scenario: 2 VRegs, both live at pressure point.
        // VReg 0: depth=0 (outside loop). VReg 1: depth=2 (inside loop).
        // Both have next-use at infinity. The one with lower depth should be spilled.
        let insts = vec![
            iconst_inst(0, 10),
            use_inst(2, 0), // v0 used here
            iconst_inst(1, 20),
            use_inst(3, 1), // v1 used here
        ];

        // Manually create a liveness info where both v0 and v1 are live at inst 0.
        let live_at: Vec<HashSet<VReg>> = vec![
            [VReg(0), VReg(1)].iter().copied().collect(), // pressure at inst 0
            [VReg(0), VReg(1)].iter().copied().collect(),
            [VReg(1)].iter().copied().collect(),
            [VReg(1)].iter().copied().collect(),
        ];
        let liveness = LivenessInfo {
            live_at,
            live_in: HashSet::new(),
            live_out: HashSet::new(),
        };

        // Both VRegs are in the interference graph (num_vregs=4).
        let graph = InterferenceGraph {
            num_vregs: 4,
            adj: vec![
                HashSet::new(),
                HashSet::new(),
                HashSet::new(),
                HashSet::new(),
            ],
            reg_class: vec![RegClass::GPR; 4],
        };

        let mut loop_depths = HashMap::new();
        loop_depths.insert(VReg(0), 0u32); // outside loop
        loop_depths.insert(VReg(1), 2u32); // inside loop (depth 2)

        // select_spill with 1 available register: must pick one of the two.
        // Due to loop penalty, VReg 1 (depth=2) should NOT be spilled.
        // VReg 0 (depth=0) should be chosen.
        let candidate = select_spill(&graph, &liveness, &insts, 1, &loop_depths);
        assert_eq!(
            candidate,
            Some(0),
            "should spill VReg 0 (outside loop), not VReg 1 (inside loop)"
        );
    }

    // Rematerialization: Iconst is re-emitted before each use, no SpillStore.
    #[test]
    fn rematerialization_no_store() {
        let mut insts = vec![
            iconst_inst(0, 42), // v0 = iconst(42)
            use_inst(1, 0),     // v1 = use(v0)
            use_inst(2, 0),     // v2 = use(v0)
        ];
        let mut spilled = HashSet::new();
        spilled.insert(0usize);
        let mut spill_slots = 0u32;
        let mut next_vreg = 10u32;

        insert_spills(
            &mut insts,
            &spilled,
            &mut spill_slots,
            &mut next_vreg,
            &HashMap::new(),
        );

        // No SpillStore: constants are rematerializable.
        assert!(
            !insts.iter().any(|i| is_spill_store(i)),
            "no SpillStore for rematerializable Iconst"
        );
        // No SpillLoad either: replaced by re-emitted Iconst.
        assert!(
            !insts.iter().any(|i| is_spill_load(i)),
            "no SpillLoad for rematerializable Iconst"
        );
        // spill_slots unchanged.
        assert_eq!(spill_slots, 0);
    }

    // Phase 4.4: XMM VReg spill inserts XMM-specific markers (not GPR markers).
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
                op: Op::Proj0,
                dst: VReg(0),
                operands: vec![VReg(99)],
            },
            use_inst(1, 0),
            use_inst(2, 0),
        ];

        let mut spilled = HashSet::new();
        spilled.insert(0usize);
        let mut spill_slots = 0u32;
        let mut next_vreg = 100u32;

        // Mark v0 as XMM class.
        let mut vreg_classes = HashMap::new();
        vreg_classes.insert(VReg(0), RegClass::XMM);

        insert_spills(
            &mut insts,
            &spilled,
            &mut spill_slots,
            &mut next_vreg,
            &vreg_classes,
        );

        // There should be an XMM SpillStore (not a GPR SpillStore).
        assert!(
            insts.iter().any(|i| is_xmm_spill_store(i)),
            "XMM VReg spill must produce XMM_SPILL_STORE_TYPE marker"
        );
        assert!(
            !insts.iter().any(|i| is_spill_store(i)),
            "XMM VReg spill must NOT produce GPR SPILL_STORE_TYPE marker"
        );

        // There should be XMM SpillLoads before each use.
        let xmm_load_count = insts.iter().filter(|i| is_xmm_spill_load(i)).count();
        assert_eq!(
            xmm_load_count, 2,
            "two XMM SpillLoads expected (one per use)"
        );
        assert!(
            !insts.iter().any(|i| is_spill_load(i)),
            "XMM VReg spill must NOT produce GPR SPILL_LOAD_TYPE marker"
        );
    }
}
