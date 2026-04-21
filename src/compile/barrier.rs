use std::collections::BTreeMap;

use crate::compile::program_point::ProgramPoint;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::BasicBlock;
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
    block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    vreg_to_arg: &mut BTreeMap<VReg, usize>,
) {
    if let Some(EffectfulOp::Branch { cond, .. }) = terminator {
        let canon = egraph.unionfind.find_immutable(*cond);
        let point = ProgramPoint::block_entry(block_idx);
        if let Some(vreg) = class_to_vreg.lookup(canon, point) {
            // Force the cond VReg into the group after all effectful ops.
            // Use max (not min like mark_arg) because we need this to come
            // AFTER all calls, overriding any earlier constraint.
            let entry = vreg_to_arg.entry(vreg).or_insert(non_term_count);
            *entry = (*entry).max(non_term_count);
        }
    }
}

/// Build barrier maps and mark the branch condition in one step.
///
/// Combines `build_barrier_maps` + `mark_branch_cond_barrier` which are always
/// called together.
pub(super) fn build_barrier_context(
    block: &BasicBlock,
    block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
) -> (BTreeMap<VReg, usize>, BTreeMap<VReg, usize>) {
    let non_term_count = block.non_term_count();
    let non_term_ops = &block.ops[..non_term_count];
    let (result_map, mut arg_map) =
        build_barrier_maps(non_term_ops, block_idx, egraph, class_to_vreg);
    mark_branch_cond_barrier(
        block.ops.last(),
        non_term_count,
        block_idx,
        egraph,
        class_to_vreg,
        &mut arg_map,
    );
    (result_map, arg_map)
}

/// Build barrier maps: which VRegs are produced/consumed by each effectful op.
pub(super) fn build_barrier_maps(
    non_term_ops: &[EffectfulOp],
    block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
) -> (BTreeMap<VReg, usize>, BTreeMap<VReg, usize>) {
    let point = ProgramPoint::block_entry(block_idx);
    let mut vreg_to_result: BTreeMap<VReg, usize> = BTreeMap::new();
    let mut vreg_to_arg: BTreeMap<VReg, usize> = BTreeMap::new();
    // Helper: mark a ClassId as consumed by barrier_k (earliest consumer wins).
    let mut mark_arg = |cid: ClassId, barrier_k: usize| {
        let canon = egraph.unionfind.find_immutable(cid);
        if let Some(vreg) = class_to_vreg.lookup(canon, point) {
            let entry = vreg_to_arg.entry(vreg).or_insert(barrier_k);
            *entry = (*entry).min(barrier_k);
        }
    };
    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        match op {
            EffectfulOp::Load { addr, result, .. } => {
                let canon = egraph.unionfind.find_immutable(*result);
                if let Some(vreg) = class_to_vreg.lookup(canon, point) {
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
                    if let Some(vreg) = class_to_vreg.lookup(canon, point) {
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
    // Propagate barrier arg constraints to transitive operands.
    // If v3 must be ready at barrier 0, then v3's operands must also be at
    // barrier 0 or earlier. Without this, an operand with a later barrier
    // constraint (e.g. v0 at barrier 4) would pull v3 to group 4 via the
    // forward pass's max(operand_groups, barrier_constraint).
    let mut vreg_to_arg = vreg_to_arg_of_barrier.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for inst in sched {
            if let Some(&barrier_k) = vreg_to_arg.get(&inst.dst) {
                for &op in &inst.operands {
                    let entry = vreg_to_arg.entry(op).or_insert(barrier_k);
                    if barrier_k < *entry {
                        *entry = barrier_k;
                        changed = true;
                    }
                }
            }
        }
    }
    let vreg_to_arg_of_barrier = &vreg_to_arg;

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
        // vreg_to_arg (which reflects consuming barriers); their operands
        // (populated by populate_effectful_operands) keep them alive.
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

        // Place SpillStore right after the def of `v` in the schedule (so it
        // sits inside `def_group`). Place SpillLoad right before the first
        // consumer in `consumer_group` (so the reload is guaranteed to execute
        // before the use even without a post-pass barrier re-sort).
        let store_inst = ScheduledInst {
            op: Op::SpillStore(slot as i64),
            dst: store_vreg,
            operands: vec![*v],
        };
        vreg_group.insert(store_vreg, *def_group);

        let load_inst = ScheduledInst {
            op: Op::SpillLoad(slot as i64),
            dst: reload_vreg,
            operands: vec![],
        };
        vreg_group.insert(reload_vreg, *consumer_group);

        // Find def_pos (after `v`'s def) and consumer_pos (before the first
        // scheduled consumer of `v`). Both are computed on the pre-insertion
        // schedule to avoid index drift.
        let def_pos = schedule
            .iter()
            .position(|inst| inst.dst == *v)
            .map(|i| i + 1)
            .unwrap_or(schedule.len());
        let consumer_pos = schedule
            .iter()
            .position(|inst| inst.operands.contains(v))
            .unwrap_or(schedule.len());

        // Insert in reverse order of position (larger index first) so the
        // earlier insertion doesn't shift the later one.
        let (first_pos, first_inst, second_pos, second_inst) = if def_pos <= consumer_pos {
            (consumer_pos, load_inst, def_pos, store_inst)
        } else {
            (def_pos, store_inst, consumer_pos, load_inst)
        };
        schedule.insert(first_pos, first_inst);
        schedule.insert(second_pos, second_inst);

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

/// Populate effectful-op operands directly onto barrier instructions in the schedule.
///
/// For Load: appends addr VReg + Addr children to the existing `LoadResult` instruction.
/// For Call with results: appends arg VRegs + Addr children to the existing `CallResult`.
/// For void Call: inserts a `VoidCallBarrier` pseudo-instruction with arg VRegs.
/// For Store: inserts a `StoreBarrier` pseudo-instruction with addr/val VRegs.
///
/// This replaces `insert_effectful_use_markers`: instead of separate EffectfulUse
/// pseudo-ops, the operands live directly on the barrier instruction itself, so
/// liveness analysis naturally sees them.
pub(super) fn populate_effectful_operands(
    schedule: &mut Vec<ScheduledInst>,
    non_term_ops: &[EffectfulOp],
    block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
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

    // Collect markers to insert (for Store and void Call only).
    let mut markers: Vec<(usize, ScheduledInst)> = Vec::new();

    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        // Program point for this barrier: used for point-aware VReg lookup.
        let barrier_pt = ProgramPoint::barrier_point(block_idx, barrier_k, schedule);

        // Resolve ClassIds to VRegs with Addr children, dedup.
        let resolve_vregs = |cids: &[ClassId], point: ProgramPoint| -> Vec<VReg> {
            let mut vregs = Vec::new();
            for &cid in cids {
                let canon = egraph.unionfind.find_immutable(cid);
                let Some(vreg) = class_to_vreg.lookup(canon, point) else {
                    continue;
                };
                vregs.push(vreg);
                if let Some(children) = addr_children.get(&vreg) {
                    vregs.extend_from_slice(children);
                }
            }
            vregs.sort_by_key(|v| v.0);
            vregs.dedup();
            vregs
        };

        match op {
            EffectfulOp::Load { addr, result, .. } => {
                let cids = [*addr];
                let vregs = resolve_vregs(&cids, barrier_pt);
                if vregs.is_empty() {
                    continue;
                }
                // Find the LoadResult instruction by its result VReg.
                let result_canon = egraph.unionfind.find_immutable(*result);
                let Some(result_vreg) = class_to_vreg.lookup(result_canon, barrier_pt) else {
                    continue;
                };
                if let Some(inst) = schedule.iter_mut().find(|i| i.dst == result_vreg) {
                    inst.operands.extend(vregs);
                    // Dedup after extending (in case of overlap with existing operands).
                    inst.operands.sort_by_key(|v| v.0);
                    inst.operands.dedup();
                }
            }
            EffectfulOp::Call { args, results, .. } => {
                let vregs = resolve_vregs(args, barrier_pt);
                if let Some(first_result) = results.first() {
                    // Non-void call: attach to existing CallResult.
                    if vregs.is_empty() {
                        continue;
                    }
                    let result_canon = egraph.unionfind.find_immutable(*first_result);
                    let Some(result_vreg) = class_to_vreg.lookup(result_canon, barrier_pt) else {
                        continue;
                    };
                    if let Some(inst) = schedule.iter_mut().find(|i| i.dst == result_vreg) {
                        inst.operands.extend(vregs);
                        inst.operands.sort_by_key(|v| v.0);
                        inst.operands.dedup();
                    }
                } else {
                    // Void call: always insert VoidCallBarrier, even with no
                    // arg VRegs. The barrier is needed as a call-clobber marker
                    // so the register allocator sees the call point.
                    let dst = VReg(*next_vreg);
                    *next_vreg += 1;
                    markers.push((
                        barrier_k,
                        ScheduledInst {
                            op: Op::VoidCallBarrier,
                            dst,
                            operands: vregs,
                        },
                    ));
                }
            }
            EffectfulOp::Store { addr, val, .. } => {
                let cids = [*addr, *val];
                let vregs = resolve_vregs(&cids, barrier_pt);
                if vregs.is_empty() {
                    continue;
                }
                let dst = VReg(*next_vreg);
                *next_vreg += 1;
                markers.push((
                    barrier_k,
                    ScheduledInst {
                        op: Op::StoreBarrier,
                        dst,
                        operands: vregs,
                    },
                ));
            }
            _ => continue,
        }
    }

    // Insert markers (StoreBarrier, VoidCallBarrier) at the correct positions.
    // Each marker goes at the END of its barrier group (same logic as EffectfulUse).
    markers.reverse();
    for (barrier_k, marker) in markers {
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
