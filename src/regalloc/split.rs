use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::Op;
use crate::ir::types::Type;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::RegClass;

use super::global_liveness::GlobalLiveness;

/// Maps cross-block VRegs to dedicated spill slots.
pub struct CrossBlockSpillMap {
    /// Spill slot assigned to each cross-block VReg that requires a stack slot.
    pub vreg_to_slot: BTreeMap<VReg, u32>,
    /// VRegs that can be rematerialized (defined by Iconst) instead of spilled.
    pub remat_vregs: BTreeSet<VReg>,
    /// Total number of cross-block spill slots allocated.
    pub num_slots: u32,
}

/// Assign cross-block spill slots for values that flow across block boundaries.
///
/// For each VReg that appears in any block's live_in (meaning it was defined
/// in a different block), assign a dedicated spill slot -- unless it is
/// rematerializable (Iconst), in which case it is flagged for rematerialization.
///
/// `block_param_vregs[i]` contains VRegs that are block parameters for block i;
/// these are handled by phi elimination and should not get cross-block slots.
pub fn assign_cross_block_slots(
    global_liveness: &GlobalLiveness,
    block_schedules: &[Vec<ScheduledInst>],
    block_param_vregs: &[BTreeSet<VReg>],
) -> CrossBlockSpillMap {
    // Build a map from VReg to its defining instruction (for rematerialization check).
    let mut def_inst: BTreeMap<VReg, &ScheduledInst> = BTreeMap::new();
    for sched in block_schedules {
        for inst in sched {
            def_inst.entry(inst.dst).or_insert(inst);
        }
    }

    // Collect all VRegs that cross block boundaries: appear in some block's live_in.
    let mut cross_block_vregs: BTreeSet<VReg> = BTreeSet::new();
    for (b, live_in) in global_liveness.live_in.iter().enumerate() {
        let params = if b < block_param_vregs.len() {
            &block_param_vregs[b]
        } else {
            // No param set for this block, treat as empty.
            &BTreeSet::new()
        };
        for &v in live_in {
            if !params.contains(&v) {
                cross_block_vregs.insert(v);
            }
        }
    }

    let mut vreg_to_slot: BTreeMap<VReg, u32> = BTreeMap::new();
    let mut remat_vregs: BTreeSet<VReg> = BTreeSet::new();
    let mut num_slots = 0u32;

    for v in &cross_block_vregs {
        if let Some(inst) = def_inst.get(v) {
            if is_rematerializable_inst(inst) {
                remat_vregs.insert(*v);
            } else {
                let slot = num_slots;
                num_slots += 1;
                vreg_to_slot.insert(*v, slot);
            }
        } else {
            // VReg has no known defining instruction (e.g., function param, block param).
            // These should not need cross-block slots since they are handled separately.
            // Skip them.
        }
    }

    CrossBlockSpillMap {
        vreg_to_slot,
        remat_vregs,
        num_slots,
    }
}

/// Returns true if the instruction is rematerializable (can be cheaply re-emitted).
/// Iconst and StackAddr have no dependencies and produce a constant value,
/// so they can be re-emitted in any block without spilling.
fn is_rematerializable_inst(inst: &ScheduledInst) -> bool {
    matches!(&inst.op, Op::Iconst(_, _) | Op::StackAddr(_))
}

/// Rewrite a single block's schedule to insert cross-block spill/reload code.
///
/// For each VReg in live_in(block) that is:
///   - NOT a block parameter (phi dest)
///   - NOT defined in this block
///
///   This function inserts:
///     * If rematerializable: a fresh Iconst instruction defining a new VReg
///     * Otherwise: a SpillLoad from the assigned cross-block slot
///
///   And rewrites all uses of the original VReg in this block to use the new one.
///
/// For each VReg in live_out(block) that is defined in this block:
///   - If NOT rematerializable: inserts a SpillStore after the block's instructions
///
/// Pass-through optimization: a VReg that is live_in AND live_out of a block but
/// NOT used or defined in the block (pure pass-through) is skipped entirely.
///
/// Returns the rewritten schedule and a rename map (old_vreg -> new_reload_vreg).
pub fn rewrite_block_for_splitting(
    schedule: &[ScheduledInst],
    block_idx: usize,
    global_liveness: &GlobalLiveness,
    spill_map: &CrossBlockSpillMap,
    block_defs: &[BTreeSet<VReg>],
    def_insts: &BTreeMap<VReg, ScheduledInst>,
    next_vreg: &mut u32,
    block_params: &BTreeSet<VReg>,
    vreg_classes: &BTreeMap<VReg, RegClass>,
) -> (Vec<ScheduledInst>, BTreeMap<VReg, VReg>) {
    let live_in = &global_liveness.live_in[block_idx];
    let live_out = &global_liveness.live_out[block_idx];
    let defs_in_block = &block_defs[block_idx];

    // Determine which VRegs are used or defined in this block (for pass-through detection).
    let used_in_block: BTreeSet<VReg> = schedule
        .iter()
        .flat_map(|inst| inst.operands.iter().copied())
        .collect();

    // Build a rename map: original VReg -> fresh VReg (for reloads/remats at block entry).
    let mut rename: BTreeMap<VReg, VReg> = BTreeMap::new();
    let mut entry_insts: Vec<ScheduledInst> = Vec::new();

    for &v in live_in {
        // Skip block parameters: handled by phi elimination.
        if block_params.contains(&v) {
            continue;
        }
        // Skip VRegs defined in this block (they don't need reloading).
        if defs_in_block.contains(&v) {
            continue;
        }

        // Pass-through optimization: if v is live_in AND live_out but not used in
        // this block's schedule, skip it entirely (stays in its slot).
        if live_out.contains(&v) && !used_in_block.contains(&v) {
            continue;
        }

        // Insert entry reload/remat.
        if spill_map.remat_vregs.contains(&v) {
            // Rematerializable: re-emit the defining instruction with a fresh VReg.
            if let Some(orig_inst) = def_insts.get(&v) {
                let new_vreg = VReg(*next_vreg);
                *next_vreg += 1;
                entry_insts.push(ScheduledInst {
                    op: orig_inst.op.clone(),
                    dst: new_vreg,
                    operands: orig_inst.operands.clone(),
                });
                rename.insert(v, new_vreg);
            }
        } else if let Some(&slot) = spill_map.vreg_to_slot.get(&v) {
            // Non-rematerializable: insert a SpillLoad from the cross-block slot.
            let new_vreg = VReg(*next_vreg);
            *next_vreg += 1;
            let is_xmm = vreg_classes
                .get(&v)
                .copied()
                .map(|c| c == RegClass::XMM)
                .unwrap_or(false);
            let load_op = if is_xmm {
                Op::XmmSpillLoad(slot as i64)
            } else {
                Op::SpillLoad(slot as i64)
            };
            entry_insts.push(ScheduledInst {
                op: load_op,
                dst: new_vreg,
                operands: vec![],
            });
            rename.insert(v, new_vreg);
        }
        // If v is in live_in but has no slot and is not remat, it might be a function
        // param or block param that wasn't filtered. Skip it silently.
    }

    // Rewrite the block's instruction list: apply the rename map to operands.
    let mut rewritten: Vec<ScheduledInst> = Vec::with_capacity(entry_insts.len() + schedule.len());
    rewritten.extend(entry_insts);

    for inst in schedule {
        let new_operands: Vec<VReg> = inst
            .operands
            .iter()
            .map(|&op| rename.get(&op).copied().unwrap_or(op))
            .collect();
        rewritten.push(ScheduledInst {
            op: inst.op.clone(),
            dst: inst.dst,
            operands: new_operands,
        });
    }

    // Insert exit spills: for each VReg in live_out(block) that is defined in this block,
    // append a SpillStore (unless rematerializable -- successors will re-emit it).
    for &v in live_out {
        if !defs_in_block.contains(&v) {
            continue;
        }
        // Phi source VRegs that are cross-block live-out need spilling.
        // Rematerializable values don't need a spill.
        if spill_map.remat_vregs.contains(&v) {
            continue;
        }
        if let Some(&slot) = spill_map.vreg_to_slot.get(&v) {
            let is_xmm = vreg_classes
                .get(&v)
                .copied()
                .map(|c| c == RegClass::XMM)
                .unwrap_or(false);
            let store_op = if is_xmm {
                Op::XmmSpillStore(slot as i64)
            } else {
                Op::SpillStore(slot as i64)
            };
            // Dummy dst VReg for the store instruction.
            let dummy_dst = VReg(*next_vreg);
            *next_vreg += 1;
            rewritten.push(ScheduledInst {
                op: store_op,
                dst: dummy_dst,
                operands: vec![v],
            });
        }
    }

    (rewritten, rename)
}

/// Compute def sets per block: which VRegs are defined in each block's schedule.
pub fn compute_block_defs(block_schedules: &[Vec<ScheduledInst>]) -> Vec<BTreeSet<VReg>> {
    block_schedules
        .iter()
        .map(|sched| sched.iter().map(|inst| inst.dst).collect())
        .collect()
}

/// Build a vreg_classes map from block schedules, identifying XMM VRegs.
pub fn build_vreg_classes_from_schedules(
    block_schedules: &[Vec<ScheduledInst>],
) -> BTreeMap<VReg, RegClass> {
    let mut map: BTreeMap<VReg, RegClass> = BTreeMap::new();

    for sched in block_schedules {
        for inst in sched {
            let class = if is_fp_op(&inst.op) {
                RegClass::XMM
            } else {
                RegClass::GPR
            };
            map.insert(inst.dst, class);
            for &op in &inst.operands {
                map.entry(op).or_insert(RegClass::GPR);
            }
        }
    }

    // Propagate XMM class: if an instruction is FP, its operands are also XMM.
    for sched in block_schedules {
        for inst in sched {
            if is_fp_op(&inst.op) {
                for &op in &inst.operands {
                    map.insert(op, RegClass::XMM);
                }
            }
        }
    }

    map
}

fn is_fp_op(op: &Op) -> bool {
    match op {
        // F64 arithmetic
        Op::X86Addsd
        | Op::X86Subsd
        | Op::X86Mulsd
        | Op::X86Divsd
        | Op::X86Sqrtsd
        // F32 arithmetic
        | Op::X86Addss
        | Op::X86Subss
        | Op::X86Mulss
        | Op::X86Divss
        | Op::X86Sqrtss
        // Conversions that produce XMM results
        | Op::X86Cvtsi2sd
        | Op::X86Cvtsi2ss
        | Op::X86Cvtsd2ss
        | Op::X86Cvtss2sd
        // FP constants
        | Op::Fconst(_, _)
        // XMM spill reloads produce XMM values
        | Op::XmmSpillLoad(_) => true,
        // Block parameters (phi destinations) with float types
        Op::BlockParam(_, _, ty) => ty.is_float(),
        // Call results with float return types
        Op::CallResult(_, ty) => ty.is_float(),
        // Function parameters with float types
        Op::Param(_, ty) => ty.is_float(),
        Op::X86Bitcast { to, .. } => matches!(to, Type::F32 | Type::F64),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc::global_liveness::compute_global_liveness;
    use crate::regalloc::spill::{is_spill_load, is_spill_store, spill_slot_of};

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

    #[allow(dead_code)]
    fn add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::X86Add,
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    fn empty_params(n: usize) -> Vec<BTreeSet<VReg>> {
        vec![BTreeSet::new(); n]
    }

    fn make_def_insts(schedules: &[Vec<ScheduledInst>]) -> BTreeMap<VReg, ScheduledInst> {
        let mut map = BTreeMap::new();
        for sched in schedules {
            for inst in sched {
                map.entry(inst.dst).or_insert_with(|| inst.clone());
            }
        }
        map
    }

    // Test 1: Single cross-block non-remat value gets a slot.
    // Block 0 defines v0 (non-remat), block 1 uses it.
    // Rewrite block 0: should insert SpillStore for v0.
    // Rewrite block 1: should insert SpillLoad for v0 -> v_new.
    #[test]
    fn cross_block_non_remat_spills() {
        // Block 0: v0 = proj0(v99) -- non-remat
        // Block 1: v1 = use(v0)
        let schedules = vec![
            vec![ScheduledInst {
                op: Op::Proj0,
                dst: VReg(0),
                operands: vec![VReg(99)],
            }],
            vec![use_inst(1, 0)],
        ];
        let successors = vec![vec![1usize], vec![]];
        let phi_uses = empty_params(2);
        let params = empty_params(2);

        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);
        let block_defs = compute_block_defs(&schedules);
        let spill_map = assign_cross_block_slots(&gl, &schedules, &params);
        let def_insts = make_def_insts(&schedules);
        let vreg_classes = BTreeMap::new();

        // v0 should get a slot (not rematerializable).
        assert!(spill_map.vreg_to_slot.contains_key(&VReg(0)));
        assert_eq!(spill_map.num_slots, 1);

        let mut next_vreg = 100u32;

        // Rewrite block 0 (defining block): should insert a SpillStore at exit.
        let (block0_rewritten, _rename0) = rewrite_block_for_splitting(
            &schedules[0],
            0,
            &gl,
            &spill_map,
            &block_defs,
            &def_insts,
            &mut next_vreg,
            &params[0],
            &vreg_classes,
        );
        let stores: Vec<_> = block0_rewritten
            .iter()
            .filter(|i| is_spill_store(i))
            .collect();
        assert_eq!(stores.len(), 1, "one SpillStore for v0 at block 0 exit");
        assert_eq!(spill_slot_of(stores[0]), 0);

        // Rewrite block 1 (using block): should insert a SpillLoad at entry.
        let (block1_rewritten, rename1) = rewrite_block_for_splitting(
            &schedules[1],
            1,
            &gl,
            &spill_map,
            &block_defs,
            &def_insts,
            &mut next_vreg,
            &params[1],
            &vreg_classes,
        );
        let loads: Vec<_> = block1_rewritten
            .iter()
            .filter(|i| is_spill_load(i))
            .collect();
        assert_eq!(loads.len(), 1, "one SpillLoad for v0 at block 1 entry");

        // The rename map should contain v0 -> new_vreg.
        assert!(rename1.contains_key(&VReg(0)));
        let new_v = rename1[&VReg(0)];
        // The use instruction in block 1 should now use new_v instead of v0.
        let use_inst_rewritten = block1_rewritten.iter().find(|i| i.dst == VReg(1)).unwrap();
        assert!(
            use_inst_rewritten.operands.contains(&new_v),
            "use of v0 should be rewritten to new reload VReg"
        );
    }

    // Test 2: Rematerializable (Iconst) cross-block value.
    // No SpillStore/SpillLoad; instead, the Iconst is re-emitted.
    #[test]
    fn cross_block_remat_no_spill() {
        // Block 0: v0 = iconst(42)  [rematerializable]
        // Block 1: v1 = use(v0)
        let schedules = vec![vec![iconst_inst(0, 42)], vec![use_inst(1, 0)]];
        let successors = vec![vec![1usize], vec![]];
        let phi_uses = empty_params(2);
        let params = empty_params(2);

        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);
        let block_defs = compute_block_defs(&schedules);
        let spill_map = assign_cross_block_slots(&gl, &schedules, &params);
        let def_insts = make_def_insts(&schedules);
        let vreg_classes = BTreeMap::new();

        // v0 should be rematerializable, no slot.
        assert!(spill_map.remat_vregs.contains(&VReg(0)));
        assert_eq!(spill_map.num_slots, 0);

        let mut next_vreg = 100u32;

        // Rewrite block 0: no SpillStore (rematerializable).
        let (block0_rewritten, _) = rewrite_block_for_splitting(
            &schedules[0],
            0,
            &gl,
            &spill_map,
            &block_defs,
            &def_insts,
            &mut next_vreg,
            &params[0],
            &vreg_classes,
        );
        assert!(
            !block0_rewritten.iter().any(|i| is_spill_store(i)),
            "no SpillStore for rematerializable Iconst"
        );

        // Rewrite block 1: should re-emit the Iconst (not a SpillLoad).
        let (block1_rewritten, rename1) = rewrite_block_for_splitting(
            &schedules[1],
            1,
            &gl,
            &spill_map,
            &block_defs,
            &def_insts,
            &mut next_vreg,
            &params[1],
            &vreg_classes,
        );
        assert!(
            !block1_rewritten.iter().any(|i| is_spill_load(i)),
            "no SpillLoad for rematerializable Iconst"
        );
        // There should be a re-emitted Iconst.
        let new_v = rename1[&VReg(0)];
        assert!(
            block1_rewritten
                .iter()
                .any(|i| i.dst == new_v && matches!(&i.op, Op::Iconst(42, _))),
            "Iconst re-emitted in block 1 with correct value"
        );
    }

    // Test 3: Pass-through value (live_in AND live_out but not used in block).
    // No reload/spill should be inserted for that block.
    #[test]
    fn pass_through_value_skipped() {
        // Block 0: defines v0
        // Block 1: pass-through (v0 not used, just flows to block 2)
        // Block 2: uses v0
        let schedules = vec![
            vec![ScheduledInst {
                op: Op::Proj0,
                dst: VReg(0),
                operands: vec![VReg(99)],
            }],
            vec![iconst_inst(1, 2)], // block 1: no use of v0
            vec![use_inst(2, 0)],    // block 2: uses v0
        ];
        let successors = vec![vec![1usize], vec![2], vec![]];
        let phi_uses = empty_params(3);
        let params = empty_params(3);

        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);
        let block_defs = compute_block_defs(&schedules);
        let spill_map = assign_cross_block_slots(&gl, &schedules, &params);
        let def_insts = make_def_insts(&schedules);
        let vreg_classes = BTreeMap::new();

        // v0 is in live_in[1] AND live_out[1] (pass-through).
        assert!(gl.live_in[1].contains(&VReg(0)));
        assert!(gl.live_out[1].contains(&VReg(0)));

        let mut next_vreg = 100u32;

        // Rewrite block 1: pass-through, no reload or spill for v0.
        let (block1_rewritten, rename1) = rewrite_block_for_splitting(
            &schedules[1],
            1,
            &gl,
            &spill_map,
            &block_defs,
            &def_insts,
            &mut next_vreg,
            &params[1],
            &vreg_classes,
        );
        // No SpillLoad in block 1 for v0 (pass-through optimization).
        assert!(
            !block1_rewritten.iter().any(|i| is_spill_load(i)),
            "pass-through block should not reload v0"
        );
        assert!(
            !rename1.contains_key(&VReg(0)),
            "pass-through block should not rename v0"
        );
        // No SpillStore either (v0 is not defined in block 1).
        assert!(
            !block1_rewritten.iter().any(|i| is_spill_store(i)),
            "pass-through block should not spill v0"
        );
    }

    // Test 4: Block params are excluded from reload insertion.
    #[test]
    fn block_params_not_reloaded() {
        // Block 1 receives v5 as a block parameter.
        // It should not insert a SpillLoad for v5 even if v5 is in live_in[1].
        let schedules = vec![
            vec![iconst_inst(0, 1)],
            vec![use_inst(1, 5)], // uses v5 (block param)
        ];
        let successors = vec![vec![1usize], vec![]];
        let _phi_uses = empty_params(2);

        // Mark v5 as a block param for block 1.
        let mut params = empty_params(2);
        params[1].insert(VReg(5));

        let mut phi_uses_with_v5 = empty_params(2);
        phi_uses_with_v5[0].insert(VReg(5)); // v5 used in block 0's jump args

        // We need v5 to be in live_in[1]. Simulate: add v5 to live_in manually
        // by making block 1 use v5 (it's in operands).
        let gl = compute_global_liveness(&schedules, &successors, &phi_uses_with_v5);

        // v5 used in block 1 but not defined anywhere in schedules -> upward-exposed.
        assert!(
            gl.live_in[1].contains(&VReg(5)),
            "v5 should be live_in[1] since block 1 uses it"
        );

        let block_defs = compute_block_defs(&schedules);
        let spill_map = assign_cross_block_slots(&gl, &schedules, &params);
        let def_insts = make_def_insts(&schedules);
        let vreg_classes = BTreeMap::new();

        let mut next_vreg = 100u32;

        // v5 is a block param for block 1, so it should NOT get a cross-block slot.
        assert!(
            !spill_map.vreg_to_slot.contains_key(&VReg(5)),
            "block params should not get cross-block slots"
        );

        let (block1_rewritten, _) = rewrite_block_for_splitting(
            &schedules[1],
            1,
            &gl,
            &spill_map,
            &block_defs,
            &def_insts,
            &mut next_vreg,
            &params[1],
            &vreg_classes,
        );

        // No SpillLoad for v5 (it's a block param).
        assert!(
            !block1_rewritten.iter().any(|i| is_spill_load(i)),
            "block param v5 should not be reloaded"
        );
    }
}
