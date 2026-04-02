use std::collections::BTreeMap;

use crate::egraph::egraph::EGraph;
use crate::egraph::extract::VReg;
use crate::ir::effectful::EffectfulOp;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::schedule::scheduler::ScheduledInst;

/// If the block terminator is a Branch, mark its `cond` VReg as consumed after
/// all non-terminator barriers. This ensures the flags-setting instruction
/// (e.g. X86Sub) is scheduled in the last barrier group, after all calls that
/// would clobber EFLAGS.
pub(super) fn mark_branch_cond_barrier(
    terminator: Option<&EffectfulOp>,
    non_term_count: usize,
    egraph: &EGraph,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    vreg_to_arg: &mut BTreeMap<VReg, usize>,
) {
    if let Some(EffectfulOp::Branch { cond, .. }) = terminator {
        let canon = egraph.unionfind.find_immutable(*cond);
        if let Some(&vreg) = class_to_vreg.get(&canon) {
            // Force the cond VReg into the group after all effectful ops.
            // Use max (not min like mark_arg) because we need this to come
            // AFTER all calls, overriding any earlier constraint.
            let entry = vreg_to_arg.entry(vreg).or_insert(non_term_count);
            *entry = (*entry).max(non_term_count);
        }
    }
}

/// Build barrier maps: which VRegs are produced/consumed by each effectful op.
pub(super) fn build_barrier_maps(
    non_term_ops: &[EffectfulOp],
    egraph: &EGraph,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
) -> (BTreeMap<VReg, usize>, BTreeMap<VReg, usize>) {
    let mut vreg_to_result: BTreeMap<VReg, usize> = BTreeMap::new();
    let mut vreg_to_arg: BTreeMap<VReg, usize> = BTreeMap::new();
    // Helper: mark a ClassId as consumed by barrier_k (earliest consumer wins).
    let mut mark_arg = |cid: ClassId, barrier_k: usize| {
        let canon = egraph.unionfind.find_immutable(cid);
        if let Some(&vreg) = class_to_vreg.get(&canon) {
            let entry = vreg_to_arg.entry(vreg).or_insert(barrier_k);
            *entry = (*entry).min(barrier_k);
        }
    };
    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        match op {
            EffectfulOp::Load { addr, result, .. } => {
                let canon = egraph.unionfind.find_immutable(*result);
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    vreg_to_result.insert(vreg, barrier_k);
                }
                mark_arg(*addr, barrier_k);
            }
            EffectfulOp::Store { addr, val, .. } => {
                mark_arg(*addr, barrier_k);
                mark_arg(*val, barrier_k);
            }
            EffectfulOp::Call { args, results, .. } => {
                for &result_cid in results {
                    let canon = egraph.unionfind.find_immutable(result_cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        vreg_to_result.insert(vreg, barrier_k);
                    }
                }
                for &arg_cid in args {
                    mark_arg(arg_cid, barrier_k);
                }
            }
            _ => {}
        }
    }
    (vreg_to_result, vreg_to_arg)
}

/// Assign each scheduled instruction to a barrier group and return the group mapping.
pub(super) fn assign_barrier_groups(
    sched: &[ScheduledInst],
    vreg_to_result_of_barrier: &BTreeMap<VReg, usize>,
    vreg_to_arg_of_barrier: &BTreeMap<VReg, usize>,
) -> BTreeMap<VReg, usize> {
    let mut vreg_group: BTreeMap<VReg, usize> = BTreeMap::new();
    for inst in sched {
        let mut min_group: usize = 0;
        for &operand_vreg in &inst.operands {
            if let Some(&barrier_k) = vreg_to_result_of_barrier.get(&operand_vreg) {
                min_group = min_group.max(barrier_k + 1);
            }
            if let Some(&og) = vreg_group.get(&operand_vreg) {
                min_group = min_group.max(og);
            }
        }
        // Barrier results (LoadResult, CallResult) are anchored to the group
        // right after their producing barrier. They must NOT be pushed later by
        // vreg_to_arg (which reflects consuming barriers); EffectfulUse markers
        // handle keeping them alive until their consumers.
        if let Some(&barrier_k) = vreg_to_result_of_barrier.get(&inst.dst) {
            min_group = min_group.max(barrier_k + 1);
        } else if let Some(&arg_barrier_k) = vreg_to_arg_of_barrier.get(&inst.dst) {
            min_group = min_group.max(arg_barrier_k);
        }
        vreg_group.insert(inst.dst, min_group);
    }
    // Backward propagation: pull definitions closer to their consumers to
    // reduce register pressure. A value in group 0 consumed only in group 3
    // can move to group 3, keeping its register live for less time.
    //
    // Build consumers map: for each VReg, which scheduled instructions use it.
    let mut consumers: BTreeMap<VReg, Vec<VReg>> = BTreeMap::new();
    for inst in sched {
        for &op in &inst.operands {
            consumers.entry(op).or_default().push(inst.dst);
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for inst in sched.iter().rev() {
            let v = inst.dst;

            // Skip barrier results (LoadResult, CallResult): they are anchored
            // to the group right after their producing barrier. Moving them later
            // would let the regalloc reuse their register before consumers read it.
            if vreg_to_result_of_barrier.contains_key(&v) {
                continue;
            }

            let current = *vreg_group.get(&v).unwrap_or(&0);

            // Compute latest valid group: minimum of all consumers' groups.
            let max_from_consumers = consumers
                .get(&v)
                .and_then(|cs| cs.iter().filter_map(|c| vreg_group.get(c)).min().copied());

            // If no scheduled consumers, this VReg is only used by barriers
            // or terminators -- keep it at the forward-pass group.
            let Some(latest) = max_from_consumers else {
                continue;
            };

            // Cap: never move past a barrier that consumes this VReg.
            let cap = vreg_to_arg_of_barrier
                .get(&v)
                .copied()
                .unwrap_or(usize::MAX);
            let target = latest.min(cap);

            // Only increase (move later); never decrease below forward-pass minimum.
            if target > current {
                vreg_group.insert(v, target);
                changed = true;
            }
        }
    }
    vreg_group
}

/// Insert early spill/reload pairs for LoadResult/CallResult VRegs whose earliest
/// scheduled consumer is 2+ barrier groups away. This shortens the live range of
/// the barrier result's register, reducing pressure across intermediate groups.
///
/// Only spills barrier results consumed exclusively by pure scheduled ops (not by
/// later effectful ops), to avoid a second rename layer through effectful lowering.
///
/// Returns a rename map (original barrier-result VReg -> reload VReg) for callers
/// that need to update effectful op lookups.
pub(super) fn insert_early_barrier_spills(
    schedule: &mut Vec<ScheduledInst>,
    vreg_to_result_of_barrier: &BTreeMap<VReg, usize>,
    vreg_to_arg_of_barrier: &BTreeMap<VReg, usize>,
    vreg_group: &mut BTreeMap<VReg, usize>,
    vreg_types: &BTreeMap<VReg, Type>,
    next_vreg: &mut u32,
    spill_slot_counter: &mut u32,
) {
    // Build consumers map: for each VReg, the dst VRegs that use it as an operand.
    let mut consumers: BTreeMap<VReg, Vec<VReg>> = BTreeMap::new();
    for inst in schedule.iter() {
        for &op in &inst.operands {
            consumers.entry(op).or_default().push(inst.dst);
        }
    }

    // Identify candidates: barrier results with distant scheduled consumers.
    let mut candidates: Vec<(VReg, usize, usize)> = Vec::new(); // (vreg, def_group, consumer_group)
    for (&v, &barrier_k) in vreg_to_result_of_barrier {
        // Skip if this barrier result is also consumed by a later effectful op.
        // We can't rename it for effectful ops without a second rename layer.
        if vreg_to_arg_of_barrier.contains_key(&v) {
            continue;
        }

        let def_group = barrier_k + 1;

        // Find earliest scheduled consumer group.
        let earliest_consumer = consumers
            .get(&v)
            .and_then(|cs| cs.iter().filter_map(|c| vreg_group.get(c)).min().copied());

        let Some(consumer_group) = earliest_consumer else {
            continue; // no scheduled consumers
        };

        // Only spill if consumer is 2+ groups away.
        if consumer_group >= def_group + 2 {
            // Skip non-GPR types (Flags can't be spilled).
            if let Some(ty) = vreg_types.get(&v)
                && matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
            {
                candidates.push((v, def_group, consumer_group));
            }
        }
    }

    if candidates.is_empty() {
        return;
    }

    // Insert SpillStore/SpillLoad pairs.
    for (v, def_group, consumer_group) in &candidates {
        let slot = *spill_slot_counter;
        *spill_slot_counter += 1;

        // Fresh VReg for the SpillStore destination (not directly used).
        let store_vreg = VReg(*next_vreg);
        *next_vreg += 1;

        // Fresh VReg for the SpillLoad result (replaces v in consumers).
        let reload_vreg = VReg(*next_vreg);
        *next_vreg += 1;

        // Insert SpillStore in def_group (right after barrier produces the value).
        let store_inst = ScheduledInst {
            op: Op::SpillStore(slot as i64),
            dst: store_vreg,
            operands: vec![*v],
        };
        schedule.push(store_inst);
        vreg_group.insert(store_vreg, *def_group);

        // Insert SpillLoad in consumer_group.
        let load_inst = ScheduledInst {
            op: Op::SpillLoad(slot as i64),
            dst: reload_vreg,
            operands: vec![],
        };
        schedule.push(load_inst);
        vreg_group.insert(reload_vreg, *consumer_group);

        // Inherit type for the reload VReg.
        // (vreg_types is not &mut, so the caller must update it if needed.)

        // Rewrite all scheduled consumers of v to use reload_vreg instead.
        // The SpillStore (which references v as operand) must keep the original.
        for inst in schedule.iter_mut() {
            if inst.dst == store_vreg {
                continue; // don't rewrite the SpillStore's operand
            }
            for op in inst.operands.iter_mut() {
                if *op == *v {
                    *op = reload_vreg;
                }
            }
        }
    }
}

/// Insert `Op::EffectfulUse` pseudo-instructions into the schedule at barrier
/// boundaries so the register allocator sees correct liveness for effectful-op
/// operands without separate deadline/live_out workarounds.
pub(super) fn insert_effectful_use_markers(
    schedule: &mut Vec<ScheduledInst>,
    non_term_ops: &[EffectfulOp],
    egraph: &EGraph,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    vreg_group: &mut BTreeMap<VReg, usize>,
    next_vreg: &mut u32,
) {
    // Build Addr-child lookup: for each VReg that defines an Addr node,
    // record its operand children.
    let addr_children: BTreeMap<VReg, Vec<VReg>> = schedule
        .iter()
        .filter(|inst| matches!(inst.op, Op::Addr { .. }))
        .map(|inst| {
            let children: Vec<VReg> = inst
                .operands
                .iter()
                .copied()
                .filter(|v| v.0 != u32::MAX)
                .collect();
            (inst.dst, children)
        })
        .collect();

    // Build all markers first, then push them.
    let mut markers: Vec<(usize, ScheduledInst)> = Vec::new();

    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        // Collect ClassId operands for this barrier.
        let cids: Vec<ClassId> = match op {
            EffectfulOp::Store { addr, val, .. } => vec![*addr, *val],
            EffectfulOp::Load { addr, .. } => vec![*addr],
            EffectfulOp::Call { args, .. } => args.clone(),
            _ => continue,
        };

        // Resolve ClassIds to VRegs, and add Addr children.
        let mut vregs: Vec<VReg> = Vec::new();
        for cid in cids {
            let canon = egraph.unionfind.find_immutable(cid);
            let Some(&vreg) = class_to_vreg.get(&canon) else {
                continue;
            };
            vregs.push(vreg);
            // If this VReg defines an Addr node, also add its children (base/index).
            if let Some(children) = addr_children.get(&vreg) {
                vregs.extend_from_slice(children);
            }
        }

        // Deduplicate.
        vregs.sort_by_key(|v| v.0);
        vregs.dedup();

        if vregs.is_empty() {
            continue;
        }

        // Allocate a fresh dst VReg for the marker.
        let dst = VReg(*next_vreg);
        *next_vreg += 1;

        let marker = ScheduledInst {
            op: Op::EffectfulUse,
            dst,
            operands: vregs,
        };

        markers.push((barrier_k, marker));
    }

    // Insert markers at the correct positions in the already-sorted schedule.
    // Each marker goes at the END of its barrier group's instructions (right
    // before the next group starts). This keeps operands alive through the
    // barrier execution gap without disrupting existing instruction order.
    //
    // Process markers in reverse barrier order so insertion indices stay valid.
    markers.reverse();
    for (barrier_k, marker) in markers {
        // Find the insertion point: after the last instruction in group barrier_k.
        // The schedule is already sorted by group from Phase 4b.
        let insert_pos = schedule
            .iter()
            .rposition(|inst| {
                let g = vreg_group.get(&inst.dst).copied().unwrap_or(0);
                g <= barrier_k
            })
            .map(|p| p + 1)
            .unwrap_or(0);
        vreg_group.insert(marker.dst, barrier_k);
        schedule.insert(insert_pos, marker);
    }
}
