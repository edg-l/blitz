use std::collections::{BTreeMap, BTreeSet};

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

/// Compute the composite spill score for a candidate VReg.
///
/// Returns `(next_use_penalized, tiebreaker, idx)` — higher is a better spill target.
/// Used as the key for `max_by_key` in both `select_spill` and `select_spill_for_class`.
fn spill_score(
    idx: usize,
    next_use: &BTreeMap<usize, usize>,
    range_lengths: &BTreeMap<usize, usize>,
    graph: &InterferenceGraph,
    loop_depths: &BTreeMap<VReg, u32>,
) -> (u64, u64, usize) {
    let next = next_use.get(&idx).copied().unwrap_or(usize::MAX) as u64;
    let depth = loop_depths.get(&VReg(idx as u32)).copied().unwrap_or(0);
    let penalty = 10u64.saturating_pow(depth).max(1);
    let degree = graph.adj[idx].len() as u64;
    let range_len = range_lengths.get(&idx).copied().unwrap_or(1) as u64;
    let tiebreaker = (degree * range_len) / penalty;
    (next / penalty, tiebreaker, idx)
}

/// Compute the fallback spill score (no next-use info) for a candidate VReg.
///
/// Returns `(degree_range_penalized, idx)` — higher is a better spill target.
fn spill_fallback_score(
    idx: usize,
    range_lengths: &BTreeMap<usize, usize>,
    graph: &InterferenceGraph,
    loop_depths: &BTreeMap<VReg, u32>,
) -> (u64, usize) {
    let degree = graph.adj[idx].len() as u64;
    let range_len = range_lengths.get(&idx).copied().unwrap_or(1) as u64;
    let depth = loop_depths.get(&VReg(idx as u32)).copied().unwrap_or(0);
    let penalty = 10u64.saturating_pow(depth).max(1);
    ((degree * range_len) / penalty, idx)
}

/// Select a VReg to spill using a composite heuristic:
///   Primary: farthest next-use (Belady's algorithm)
///   Tiebreaker: degree * range_length (higher = more pressure relief)
///   Loop penalty divides the whole score.
///
/// VRegs in `excluded` (pre-colored/phantom) are never selected.
///
/// Returns `None` if no spill candidate is found.
pub fn select_spill(
    graph: &InterferenceGraph,
    liveness: &LivenessInfo,
    insts: &[ScheduledInst],
    available_regs: u32,
    loop_depths: &BTreeMap<VReg, u32>,
    excluded: &BTreeSet<usize>,
) -> Option<usize> {
    let range_lengths = compute_live_range_length(insts);

    // Find the instruction index where we first exceed register pressure.
    if let Some(pressure_point) = find_pressure_point(liveness, available_regs) {
        let next_use = compute_next_use(insts, pressure_point);
        let live_at_pressure = &liveness.live_at[pressure_point];

        // Consider all VRegs live at the pressure point, excluding pre-colored/phantom.
        let candidates: Vec<usize> = live_at_pressure
            .iter()
            .map(|v| v.0 as usize)
            .filter(|&idx| idx < graph.num_vregs)
            .filter(|idx| !excluded.contains(idx))
            .collect();

        // Pick the candidate with the best composite score.
        if let Some(best) = candidates
            .into_iter()
            .max_by_key(|&idx| spill_score(idx, &next_use, &range_lengths, graph, loop_depths))
        {
            return Some(best);
        }
    }

    // Fallback: coloring overestimates but no point exceeds available_regs.
    // Pick the VReg with the highest degree*range_length, penalized by loop depth.
    (0..graph.num_vregs)
        .filter(|&idx| !graph.adj[idx].is_empty())
        .filter(|idx| !excluded.contains(idx))
        .max_by_key(|&idx| spill_fallback_score(idx, &range_lengths, graph, loop_depths))
}

fn find_pressure_point(liveness: &LivenessInfo, available_regs: u32) -> Option<usize> {
    // Find the point with maximum register pressure.
    let mut best: Option<(usize, usize)> = None; // (pressure, index)
    for (i, live_set) in liveness.live_at.iter().enumerate() {
        let pressure = live_set.len();
        if pressure >= available_regs as usize && best.is_none_or(|(bp, _)| pressure > bp) {
            best = Some((pressure, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Select a spill candidate targeting a specific register class.
///
/// Finds the XMM pressure point (instruction where the most XMM vregs are
/// simultaneously live) and picks the best XMM vreg to spill at that point.
pub fn select_spill_for_class(
    graph: &InterferenceGraph,
    liveness: &LivenessInfo,
    insts: &[ScheduledInst],
    available_regs: u32,
    loop_depths: &std::collections::BTreeMap<VReg, u32>,
    excluded: &std::collections::BTreeSet<usize>,
    target_class: crate::x86::reg::RegClass,
) -> Option<usize> {
    let range_lengths = compute_live_range_length(insts);

    // Find the instruction index where the target class has maximum pressure.
    let pressure_point = {
        let mut best: Option<(usize, usize)> = None;
        for (i, live_set) in liveness.live_at.iter().enumerate() {
            let class_pressure = live_set
                .iter()
                .filter(|v| {
                    let idx = v.0 as usize;
                    idx < graph.num_vregs && graph.reg_class[idx] == target_class
                })
                .count();
            if class_pressure >= available_regs as usize
                && best.is_none_or(|(bp, _)| class_pressure > bp)
            {
                best = Some((class_pressure, i));
            }
        }
        best.map(|(_, i)| i)
    };

    if let Some(pp) = pressure_point {
        let next_use = compute_next_use(insts, pp);
        let live_at_pressure = &liveness.live_at[pp];

        let candidates: Vec<usize> = live_at_pressure
            .iter()
            .map(|v| v.0 as usize)
            .filter(|&idx| idx < graph.num_vregs)
            .filter(|idx| !excluded.contains(idx))
            .filter(|&idx| graph.reg_class[idx] == target_class)
            .collect();

        if let Some(best) = candidates
            .into_iter()
            .max_by_key(|&idx| spill_score(idx, &next_use, &range_lengths, graph, loop_depths))
        {
            return Some(best);
        }
    }

    // Fallback: pick the target-class VReg with highest degree*range_length.
    (0..graph.num_vregs)
        .filter(|&idx| graph.reg_class[idx] == target_class)
        .filter(|&idx| !graph.adj[idx].is_empty())
        .filter(|idx| !excluded.contains(idx))
        .max_by_key(|&idx| spill_fallback_score(idx, &range_lengths, graph, loop_depths))
}

fn compute_next_use(insts: &[ScheduledInst], from: usize) -> BTreeMap<usize, usize> {
    let mut next_use: BTreeMap<usize, usize> = BTreeMap::new();
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
        if matches!(inst.op, Op::CallResult(_, _) | Op::VoidCallBarrier) {
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
    spill_slots: &mut u32,
    next_vreg: &mut u32,
    vreg_classes: &BTreeMap<VReg, crate::x86::reg::RegClass>,
) -> BTreeMap<VReg, Vec<VReg>> {
    if spilled.is_empty() {
        return BTreeMap::new();
    }

    // Build a map of VReg -> defining instruction op (for rematerialization).
    let def_ops: BTreeMap<usize, ScheduledInst> = insts
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

    // Collect call-arg VRegs: these must NOT be rematerialized even if
    // their defining op is Iconst/StackAddr, because the call needs
    // the value alive at the call point.
    let call_arg_vregs = collect_call_arg_vregs(insts);

    // Assign spill slots to non-rematerializable VRegs AND call-arg VRegs
    // (call-arg constants need a spill slot because we can't remat them away).
    let mut vreg_to_slot: BTreeMap<usize, u32> = BTreeMap::new();
    for &idx in spilled {
        let is_call_arg = call_arg_vregs.contains(&idx);
        let needs_slot = if let Some(def) = def_ops.get(&idx) {
            !is_rematerializable(def) || is_call_arg
        } else {
            false
        };
        if needs_slot {
            let slot = *spill_slots;
            *spill_slots += 1;
            vreg_to_slot.insert(idx, slot);
        }
    }

    let mut reload_map: BTreeMap<VReg, Vec<VReg>> = BTreeMap::new();

    // We need to process the instruction list and insert spill/reload code.
    // We do a single pass, building a new instruction list.
    let old_insts = std::mem::take(insts);
    let mut new_insts: Vec<ScheduledInst> = Vec::with_capacity(old_insts.len() * 2);

    // Track current reload VRegs for each spilled VReg.
    // Maps original VReg index -> current reload VReg (if a reload was just inserted).
    let mut current_reload: BTreeMap<usize, VReg> = BTreeMap::new();

    for mut inst in old_insts {
        // Before this instruction, insert reloads for any spilled operands.
        let mut new_operands = Vec::with_capacity(inst.operands.len());
        for &op in &inst.operands {
            let op_idx = op.0 as usize;
            if spilled.contains(&op_idx) {
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
        let mut spilled = BTreeSet::new();
        spilled.insert(0usize); // spill v0
        let mut spill_slots = 0u32;
        let mut next_vreg = 100u32;

        insert_spills(
            &mut insts,
            &spilled,
            &mut spill_slots,
            &mut next_vreg,
            &BTreeMap::new(),
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
        let live_at: Vec<BTreeSet<VReg>> = vec![
            [VReg(0), VReg(1)].iter().copied().collect(), // pressure at inst 0
            [VReg(0), VReg(1)].iter().copied().collect(),
            [VReg(1)].iter().copied().collect(),
            [VReg(1)].iter().copied().collect(),
        ];
        let liveness = LivenessInfo {
            live_at,
            live_in: BTreeSet::new(),
            live_out: BTreeSet::new(),
        };

        // Both VRegs are in the interference graph (num_vregs=4).
        let graph = InterferenceGraph {
            num_vregs: 4,
            adj: vec![
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            ],
            reg_class: vec![RegClass::GPR; 4],
        };

        let mut loop_depths = BTreeMap::new();
        loop_depths.insert(VReg(0), 0u32); // outside loop
        loop_depths.insert(VReg(1), 2u32); // inside loop (depth 2)

        // select_spill with 1 available register: must pick one of the two.
        // Due to loop penalty, VReg 1 (depth=2) should NOT be spilled.
        // VReg 0 (depth=0) should be chosen.
        let excluded = BTreeSet::new();
        let candidate = select_spill(&graph, &liveness, &insts, 1, &loop_depths, &excluded);
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
        let mut spilled = BTreeSet::new();
        spilled.insert(0usize);
        let mut spill_slots = 0u32;
        let mut next_vreg = 10u32;

        insert_spills(
            &mut insts,
            &spilled,
            &mut spill_slots,
            &mut next_vreg,
            &BTreeMap::new(),
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

        let mut spilled = BTreeSet::new();
        spilled.insert(0usize);
        let mut spill_slots = 0u32;
        let mut next_vreg = 100u32;

        // Mark v0 as XMM class.
        let mut vreg_classes = BTreeMap::new();
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

    // Two candidates with same next-use but different degree*range. The one with
    // higher degree*range_length should be selected (more pressure relief).
    #[test]
    fn spill_prefers_high_degree_long_range() {
        use super::super::interference::InterferenceGraph;
        use super::super::liveness::LivenessInfo;
        use crate::x86::reg::RegClass;

        // v0: defined at 0, used at 4 (range=4). Degree=1.
        // v1: defined at 0, used at 4 (range=4). Degree=3.
        // Both live at pressure point (inst 0), both next-use at inst 4.
        // v1 has higher degree*range (3*4=12 vs 1*4=4), so v1 should be picked.
        let insts = vec![
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(0),
                operands: vec![],
            },
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(1),
                operands: vec![],
            },
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(2),
                operands: vec![],
            },
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(3),
                operands: vec![],
            },
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(4),
                operands: vec![VReg(0), VReg(1)],
            },
        ];

        // Both v0 and v1 live at every instruction from 0 onward.
        let live_at: Vec<BTreeSet<VReg>> = vec![
            [VReg(0), VReg(1)].iter().copied().collect(), // pressure point (size 2 >= avail 1)
            [VReg(0), VReg(1)].iter().copied().collect(),
            [VReg(0), VReg(1)].iter().copied().collect(),
            [VReg(0), VReg(1)].iter().copied().collect(),
            [VReg(0), VReg(1)].iter().copied().collect(),
        ];
        let liveness = LivenessInfo {
            live_at,
            live_in: BTreeSet::new(),
            live_out: BTreeSet::new(),
        };

        // v0: degree 1 (interferes with v1 only)
        // v1: degree 3 (interferes with v0, v2, v3)
        let mut adj = vec![BTreeSet::new(); 5];
        adj[0].insert(1);
        adj[1].insert(0);
        adj[1].insert(2);
        adj[1].insert(3);
        adj[2].insert(1);
        adj[3].insert(1);
        let graph = InterferenceGraph {
            num_vregs: 5,
            adj,
            reg_class: vec![RegClass::GPR; 5],
        };

        let loop_depths = BTreeMap::new();
        let excluded = BTreeSet::new();
        let candidate = select_spill(&graph, &liveness, &insts, 1, &loop_depths, &excluded);
        // Both have same next-use (4) and same range (4), so tiebreaker is degree*range.
        // v1: degree=3, range=4 -> 12. v0: degree=1, range=4 -> 4. v1 wins.
        assert_eq!(candidate, Some(1), "should spill v1 (higher degree*range)");
    }

    // Best candidate by score is in excluded set, verify next-best is chosen.
    #[test]
    fn spill_excludes_precolored() {
        use super::super::interference::InterferenceGraph;
        use super::super::liveness::LivenessInfo;
        use crate::x86::reg::RegClass;

        let insts = vec![
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(0),
                operands: vec![],
            },
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(1),
                operands: vec![],
            },
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(2),
                operands: vec![VReg(0), VReg(1)],
            },
        ];

        let live_at: Vec<BTreeSet<VReg>> = vec![
            [VReg(0), VReg(1)].iter().copied().collect(),
            [VReg(0), VReg(1)].iter().copied().collect(),
            [VReg(0), VReg(1)].iter().copied().collect(),
        ];
        let liveness = LivenessInfo {
            live_at,
            live_in: BTreeSet::new(),
            live_out: BTreeSet::new(),
        };

        // v0 has higher degree (better spill target), but is excluded.
        let mut adj = vec![BTreeSet::new(); 3];
        adj[0].insert(1);
        adj[0].insert(2);
        adj[1].insert(0);
        adj[2].insert(0);
        let graph = InterferenceGraph {
            num_vregs: 3,
            adj,
            reg_class: vec![RegClass::GPR; 3],
        };

        let loop_depths = BTreeMap::new();
        let mut excluded = BTreeSet::new();
        excluded.insert(0usize); // exclude v0

        let candidate = select_spill(&graph, &liveness, &insts, 1, &loop_depths, &excluded);
        assert_eq!(candidate, Some(1), "should spill v1 since v0 is excluded");
    }
}
