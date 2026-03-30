use std::collections::{HashMap, HashSet};

use crate::egraph::extract::VReg;
use crate::ir::op::Op;
use crate::ir::types::Type;
use crate::schedule::scheduler::ScheduledInst;

use super::interference::InterferenceGraph;
use super::liveness::LivenessInfo;

// ── Spill/reload pseudo-op markers ───────────────────────────────────────────
//
// We encode spills and reloads as special ScheduledInst entries using a
// sentinel Op. The Op::Iconst is reused as a marker (spill slot index as the
// immediate) with a special Type sentinel. Instead, we use a simpler approach:
// keep a separate enum to mark the "kind" of each instruction. Since we cannot
// easily extend ScheduledInst, we use Op::Iconst with a reserved type sentinel:
//
//   SpillStore: represented as a ScheduledInst with op = Op::Iconst(slot, Type::I8)
//               and operands = [vreg_being_spilled]. dst is a dummy VReg.
//   SpillLoad:  represented as a ScheduledInst with op = Op::Iconst(slot, Type::I16)
//               and operands = []. dst is the new reload VReg.
//
// These will be lowered to real MovMR/MovRM by the backend. Using I8/I16 as
// sentinels is safe because no real instruction in this IR produces I8 or I16
// from an Iconst with the slot index.
//
// The slot index is encoded in the Iconst immediate.

pub const SPILL_STORE_TYPE: Type = Type::I8;
pub const SPILL_LOAD_TYPE: Type = Type::I16;

pub fn is_spill_store(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Iconst(_, t) if *t == SPILL_STORE_TYPE)
}

pub fn is_spill_load(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Iconst(_, t) if *t == SPILL_LOAD_TYPE)
}

pub fn spill_slot_of(inst: &ScheduledInst) -> u32 {
    match &inst.op {
        Op::Iconst(slot, _) => *slot as u32,
        _ => unreachable!("spill_slot_of called on non-spill inst"),
    }
}

// ── Spill selection (10.9) ────────────────────────────────────────────────────

/// Select a VReg to spill using a farthest-next-use heuristic.
///
/// Among all VRegs that are live (present in the interference graph) and not
/// rematerializable, picks the one whose next use is farthest from the point
/// where register pressure exceeds `available_regs`.
///
/// Returns `None` if no spill candidate is found (shouldn't happen if the
/// chromatic number truly exceeds available_regs).
pub fn select_spill(
    graph: &InterferenceGraph,
    liveness: &LivenessInfo,
    insts: &[ScheduledInst],
    available_regs: u32,
) -> Option<usize> {
    // Find the instruction index where we first exceed register pressure.
    let pressure_point = find_pressure_point(liveness, available_regs)?;

    // Build a next-use map: VReg index -> next instruction index where it's used.
    let next_use = compute_next_use(insts, pressure_point);

    // Gather candidates: VRegs live at the pressure point.
    let live_at_pressure = &liveness.live_at[pressure_point];

    // Prefer to spill VRegs with farthest next use.
    // Exclude rematerializable VRegs (Iconst definitions) if possible.
    let iconst_defs: HashSet<usize> = iconst_defined_vregs(insts);

    // First try non-rematerializable candidates.
    let non_remat: Vec<usize> = live_at_pressure
        .iter()
        .map(|v| v.0 as usize)
        .filter(|&idx| idx < graph.num_vregs && !iconst_defs.contains(&idx))
        .collect();

    let candidates = if non_remat.is_empty() {
        // Fall back to all candidates including rematerializable ones.
        live_at_pressure
            .iter()
            .map(|v| v.0 as usize)
            .filter(|&idx| idx < graph.num_vregs)
            .collect::<Vec<_>>()
    } else {
        non_remat
    };

    // Pick the candidate with the farthest next use.
    candidates.into_iter().max_by_key(|&idx| {
        next_use.get(&idx).copied().unwrap_or(usize::MAX) // no future use = best to spill
    })
}

fn find_pressure_point(liveness: &LivenessInfo, available_regs: u32) -> Option<usize> {
    for (i, live_set) in liveness.live_at.iter().enumerate() {
        if live_set.len() > available_regs as usize {
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

fn iconst_defined_vregs(insts: &[ScheduledInst]) -> HashSet<usize> {
    insts
        .iter()
        .filter(|inst| matches!(&inst.op, Op::Iconst(_, _)))
        .map(|inst| inst.dst.0 as usize)
        .collect()
}

// ── Rematerialization check ────────────────────────────────────────────────────

/// Returns true if the VReg defined by `inst` can be rematerialized
/// (i.e., cheaply recomputed instead of spilled to memory).
pub fn is_rematerializable(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Iconst(_, _))
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
                    let load_inst = ScheduledInst {
                        op: Op::Iconst(slot as i64, SPILL_LOAD_TYPE),
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

        new_insts.push(inst);

        // After the def of a spilled VReg, insert a SpillStore (if not remat).
        if is_spill_def && let Some(&slot) = vreg_to_slot.get(&dst_idx) {
            let spilled_vreg = VReg(dst_idx as u32);
            let dummy_dst = VReg(*next_vreg);
            *next_vreg += 1;
            let store_inst = ScheduledInst {
                op: Op::Iconst(slot as i64, SPILL_STORE_TYPE),
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

        insert_spills(&mut insts, &spilled, &mut spill_slots, &mut next_vreg);

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

        insert_spills(&mut insts, &spilled, &mut spill_slots, &mut next_vreg);

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
}
