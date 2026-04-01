use std::collections::{HashMap, HashSet};

use crate::egraph::egraph::EGraph;
use crate::egraph::extract::VReg;
use crate::ir::effectful::EffectfulOp;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::schedule::scheduler::ScheduledInst;

/// Build barrier maps: which VRegs are produced/consumed by each effectful op.
pub(super) fn build_barrier_maps(
    non_term_ops: &[EffectfulOp],
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
) -> (HashMap<VReg, usize>, HashMap<VReg, usize>) {
    let mut vreg_to_result: HashMap<VReg, usize> = HashMap::new();
    let mut vreg_to_arg: HashMap<VReg, usize> = HashMap::new();
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
    vreg_to_result_of_barrier: &HashMap<VReg, usize>,
    vreg_to_arg_of_barrier: &HashMap<VReg, usize>,
) -> HashMap<VReg, usize> {
    let mut vreg_group: HashMap<VReg, usize> = HashMap::new();
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
        if let Some(&arg_barrier_k) = vreg_to_arg_of_barrier.get(&inst.dst) {
            min_group = min_group.max(arg_barrier_k);
        }
        vreg_group.insert(inst.dst, min_group);
    }
    // Backward propagation: pull definitions closer to their consumers to
    // reduce register pressure. A value in group 0 consumed only in group 3
    // can move to group 3, keeping its register live for less time.
    //
    // Build consumers map: for each VReg, which scheduled instructions use it.
    let mut consumers: HashMap<VReg, Vec<VReg>> = HashMap::new();
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
                .map(|cs| cs.iter().filter_map(|c| vreg_group.get(c)).min().copied())
                .flatten();

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
    vreg_to_result_of_barrier: &HashMap<VReg, usize>,
    vreg_to_arg_of_barrier: &HashMap<VReg, usize>,
    vreg_group: &mut HashMap<VReg, usize>,
    vreg_types: &HashMap<VReg, Type>,
    next_vreg: &mut u32,
    spill_slot_counter: &mut u32,
) {
    // Build consumers map: for each VReg, the dst VRegs that use it as an operand.
    let mut consumers: HashMap<VReg, Vec<VReg>> = HashMap::new();
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
            if let Some(ty) = vreg_types.get(&v) {
                if matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64) {
                    candidates.push((v, def_group, consumer_group));
                }
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

/// Compute per-VReg deadline positions for effectful op operands.
///
/// Instead of keeping all effectful operand VRegs alive until the end of the block,
/// each VReg gets a deadline: the instruction index at which it must be alive for
/// its barrier group's effectful op to execute. This reduces artificial register
/// pressure in blocks with many effectful ops.
///
/// Returns `(deadlines, covered_vregs)` where:
/// - `deadlines` maps each effectful operand VReg to its deadline instruction index
/// - `covered_vregs` is the set of VRegs that have deadlines (to remove from block_live_out)
pub(super) fn compute_effectful_deadlines(
    schedule: &[ScheduledInst],
    non_term_ops: &[EffectfulOp],
    terminator: &EffectfulOp,
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
) -> (HashMap<VReg, usize>, HashSet<VReg>) {
    if non_term_ops.is_empty() || schedule.is_empty() {
        return (HashMap::new(), HashSet::new());
    }

    // Collect VRegs used by the terminator (Ret). These must remain in
    // block_live_out even if they also appear as operands of non-terminator ops.
    let mut terminator_vregs: HashSet<VReg> = HashSet::new();
    {
        let term_cids: Vec<ClassId> = match terminator {
            EffectfulOp::Ret { val } => val.iter().copied().collect(),
            _ => vec![],
        };
        for cid in term_cids {
            let canon = egraph.unionfind.find_immutable(cid);
            if let Some(&vreg) = class_to_vreg.get(&canon) {
                terminator_vregs.insert(vreg);
            }
        }
    }

    // Build barrier maps and assign barrier groups.
    let (vreg_to_result, vreg_to_arg) = build_barrier_maps(non_term_ops, egraph, class_to_vreg);
    let vreg_group = assign_barrier_groups(schedule, &vreg_to_result, &vreg_to_arg);

    // Compute the barrier group of each instruction in the schedule.
    let inst_groups: Vec<usize> = schedule
        .iter()
        .map(|inst| *vreg_group.get(&inst.dst).unwrap_or(&0))
        .collect();

    // Find the first instruction index of each group.
    let mut group_start: HashMap<usize, usize> = HashMap::new();
    for (i, &g) in inst_groups.iter().enumerate() {
        group_start.entry(g).or_insert(i);
    }

    // Build sorted list of group starts for finding the next existing group.
    // When a barrier K has no instructions in group K+1, we use the first
    // instruction of the next group that DOES exist as the deadline.
    let mut sorted_group_starts: Vec<(usize, usize)> =
        group_start.iter().map(|(&g, &pos)| (g, pos)).collect();
    sorted_group_starts.sort_by_key(|(g, _)| *g);

    // Find the deadline position for barrier K: the first instruction of the
    // next existing group after K. Returns None if no such group exists.
    let find_deadline = |barrier_k: usize| -> Option<usize> {
        let next_group = barrier_k + 1;
        // Find the first group >= next_group that has instructions.
        sorted_group_starts
            .iter()
            .find(|(g, _)| *g >= next_group)
            .map(|(_, pos)| *pos)
    };

    // Build a map from VReg to its defining instruction, so we can find
    // transitive register dependencies through Addr nodes. The effectful
    // emission code (build_mem_addr) reads the Addr node's base/index
    // children's registers directly, not the Addr VReg's register.
    let def_inst: HashMap<VReg, &ScheduledInst> =
        schedule.iter().map(|inst| (inst.dst, inst)).collect();

    // Resolve a ClassId to (direct_vreg, addr_children).
    // The direct VReg is in block_live_out (from collect_block_effectful_operand_vregs).
    // Addr children are NOT in block_live_out but are read by build_mem_addr during emission.
    let resolve_cid = |cid: ClassId| -> (Option<VReg>, Vec<VReg>) {
        let canon = egraph.unionfind.find_immutable(cid);
        let Some(&vreg) = class_to_vreg.get(&canon) else {
            return (None, vec![]);
        };
        let mut children = vec![];
        if let Some(inst) = def_inst.get(&vreg) {
            if matches!(inst.op, Op::Addr { .. }) {
                children.extend_from_slice(&inst.operands);
            }
        }
        (Some(vreg), children)
    };

    // Collect (direct_vregs, addr_child_vregs) for an effectful op.
    let collect_op_vregs = |op: &EffectfulOp| -> (Vec<VReg>, Vec<VReg>) {
        let cids: Vec<ClassId> = match op {
            EffectfulOp::Store { addr, val, .. } => vec![*addr, *val],
            EffectfulOp::Load { addr, .. } => vec![*addr],
            EffectfulOp::Call { args, .. } => args.clone(),
            _ => return (vec![], vec![]),
        };
        let mut direct = vec![];
        let mut children = vec![];
        for cid in cids {
            let (d, c) = resolve_cid(cid);
            if let Some(v) = d {
                direct.push(v);
            }
            children.extend(c);
        }
        (direct, children)
    };

    // First pass: collect VRegs that are used by barriers without a deadline
    // (last barrier or barriers with no next-group instructions). These VRegs
    // must stay in block_live_out even if earlier barriers also use them.
    let mut must_stay_in_live_out: HashSet<VReg> = HashSet::new();
    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        if find_deadline(barrier_k).is_none() {
            let (direct, children) = collect_op_vregs(op);
            for vreg in direct.into_iter().chain(children) {
                must_stay_in_live_out.insert(vreg);
            }
        }
    }

    // Second pass: assign deadlines. Direct VRegs go into `covered` (to be
    // removed from block_live_out). Addr-child VRegs only get deadlines
    // (they were never in block_live_out, just need liveness via deadline).
    let mut deadlines: HashMap<VReg, usize> = HashMap::new();
    let mut covered: HashSet<VReg> = HashSet::new();

    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        let Some(deadline_pos) = find_deadline(barrier_k) else {
            continue;
        };

        let (direct, children) = collect_op_vregs(op);
        for vreg in direct {
            if terminator_vregs.contains(&vreg) || must_stay_in_live_out.contains(&vreg) {
                continue;
            }
            let entry = deadlines.entry(vreg).or_insert(0);
            *entry = (*entry).max(deadline_pos);
            covered.insert(vreg);
        }
        for vreg in children {
            if terminator_vregs.contains(&vreg) || must_stay_in_live_out.contains(&vreg) {
                continue;
            }
            let entry = deadlines.entry(vreg).or_insert(0);
            *entry = (*entry).max(deadline_pos);
        }
    }

    (deadlines, covered)
}
