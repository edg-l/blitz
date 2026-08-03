use std::collections::{BTreeMap, BTreeSet};

use crate::compile::program_point::ProgramPoint;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::ir::effectful::{BlockId, EffectfulOp};
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
                    // Inserting a constraint counts as a change. Without this
                    // the fixpoint can stop one level short: an operand that
                    // gains its first constraint here never gets the chance to
                    // pass it on, so a value further up the chain keeps a
                    // later barrier's constraint and is emitted after the
                    // barrier that actually needs it.
                    match vreg_to_arg.entry(op) {
                        std::collections::btree_map::Entry::Vacant(slot) => {
                            slot.insert(barrier_k);
                            changed = true;
                        }
                        std::collections::btree_map::Entry::Occupied(mut slot) => {
                            if barrier_k < *slot.get() {
                                slot.insert(barrier_k);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }
    let vreg_to_arg_of_barrier = &vreg_to_arg;

    // A division leaves its quotient in RAX and its remainder in RDX, and nothing
    // models that implicit liveness: the projection that moves a result into a
    // VReg is the only thing that reads those registers, and no interference edge
    // says so. So a projection has to run in the division's own group. What
    // separates two groups is a barrier, a call is a barrier, and a call destroys
    // both registers -- a quotient consumed after the next call read whatever that
    // call left in RAX.
    let div_dsts: BTreeSet<VReg> = sched
        .iter()
        .filter(|i| matches!(i.op, Op::X86Idiv | Op::X86Div))
        .map(|i| i.dst)
        .collect();
    let div_proj_source = |inst: &ScheduledInst| -> Option<VReg> {
        if !matches!(inst.op, Op::Proj0 | Op::Proj1) {
            return None;
        }
        inst.operands
            .first()
            .copied()
            .filter(|v| div_dsts.contains(v))
    };

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
        // Pinned, not merely bounded below: a consuming barrier's constraint would
        // otherwise push the projection past the call that clobbers the register it
        // reads.
        if let Some(src) = div_proj_source(inst)
            && let Some(&div_group) = vreg_group.get(&src)
        {
            min_group = div_group;
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

            // Division projections are anchored to their division for the same
            // reason: the register they read is not theirs to keep.
            if div_proj_source(inst).is_some() {
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

/// How many leading barrier operands carry a role, for this effectful op.
///
/// The layout `populate_effectful_operands` writes, and which every later pass
/// preserves:
///
/// | op            | operand 0 | operand 1 | operand `n..`       |
/// |---------------|-----------|-----------|---------------------|
/// | `Load`        | address   |           | folded Addr children |
/// | `Store`       | address   | value     | folded Addr children |
/// | `Call`        | arg 0 in ABI order, arg 1, ...  | folded Addr children |
///
/// A role operand is absent only when its class had no VReg at the barrier
/// point, in which case the whole list is shorter -- so a reader must bound-check
/// rather than assume the slot exists.
pub(super) fn role_operand_count(op: &EffectfulOp) -> usize {
    match op {
        EffectfulOp::Load { .. } => 1,
        EffectfulOp::Store { .. } => 2,
        EffectfulOp::Call { args, .. } => args.len(),
        EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. } => 0,
    }
}

/// Populate effectful-op operands directly onto barrier instructions in the schedule.
///
/// For Load: addr VReg, then Addr children, on the existing `LoadResult`.
/// For Call with results: arg VRegs in ABI order, then Addr children, on the
/// existing `CallResult`.
/// For void Call: inserts a `VoidCallBarrier` with the arg VRegs.
/// For Store: inserts a `StoreBarrier` with addr, val, then Addr children.
///
/// Operands live directly on the barrier instruction so liveness analysis sees
/// them without separate pseudo-ops.
///
/// # The leading operands are positional, and that is load-bearing
///
/// `role_operand_count` names how many leading operands carry a role: the
/// address of a Load, the address and value of a Store, the arguments of a Call
/// in ABI order. Anything after them (the children of a folded `Addr`) is there
/// only to keep liveness honest.
///
/// This order is never disturbed. It used to be: operands were sorted by VReg
/// index and deduped, which made the list a *set* and left Phase 7 to work out
/// which member was the address. Every heuristic that did so was a wrong-code
/// bug -- most recently a Load resolving its address to the register holding its
/// own index, `mov ecx,0x7` then `mov esi,[rcx]`. Position survives what
/// reconstruction cannot: the splitter rewrites operands by index, and
/// coalescing renames them in place, so both preserve roles for free.
///
/// Duplicates are removed only from the trailing liveness operands. A role
/// operand keeps its slot even when the same VReg fills two roles, as in
/// `*p = p`.
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
    debug_assert!(
        non_term_ops.iter().all(|op| role_operand_count(op) > 0
            || matches!(op, EffectfulOp::Call { args, .. } if args.is_empty())),
        "every non-terminator effectful op must declare its role operand count"
    );

    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        // Program point for this barrier: used for point-aware VReg lookup.
        let barrier_pt = ProgramPoint::barrier_point(block_idx, barrier_k, schedule);

        // Resolve the role ClassIds to VRegs, in order, then append the Addr
        // children they need kept alive. Roles stay at their own index; only the
        // trailing liveness operands are deduped.
        let resolve_vregs = |cids: &[ClassId], point: ProgramPoint| -> Vec<VReg> {
            let mut roles = Vec::with_capacity(cids.len());
            for &cid in cids {
                let canon = egraph.unionfind.find_immutable(cid);
                let Some(vreg) = class_to_vreg.lookup(canon, point) else {
                    // Dropping a role shifts every later one down an index, and
                    // Phase 7 reads roles by index -- a Store would take its
                    // value as its address. Every class reaching here has a VReg
                    // at the barrier point in practice; if that ever stops being
                    // true, this has to pad rather than skip.
                    debug_assert!(
                        false,
                        "barrier role for class {canon:?} has no VReg at {point:?};                          skipping it would shift the remaining roles"
                    );
                    continue;
                };
                roles.push(vreg);
            }
            let mut vregs = roles.clone();
            for role in &roles {
                if let Some(children) = addr_children.get(role) {
                    for &child in children {
                        if !vregs.contains(&child) {
                            vregs.push(child);
                        }
                    }
                }
            }
            vregs
        };

        // Append role-first operands to an existing barrier result instruction,
        // keeping the role prefix contiguous at the front.
        let append_operands = |inst: &mut ScheduledInst, vregs: Vec<VReg>| {
            let mut merged = vregs;
            for &existing in &inst.operands {
                if !merged.contains(&existing) {
                    merged.push(existing);
                }
            }
            inst.operands = merged;
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
                    append_operands(inst, vregs);
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
                        append_operands(inst, vregs);
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

/// The successor edges a terminator has, each with the arguments it passes.
pub(crate) fn terminator_edges(terminator: &EffectfulOp) -> Vec<(BlockId, &[ClassId])> {
    match terminator {
        EffectfulOp::Jump { target, args } => vec![(*target, args.as_slice())],
        EffectfulOp::Branch {
            bb_true,
            bb_false,
            true_args,
            false_args,
            ..
        } => vec![
            (*bb_true, true_args.as_slice()),
            (*bb_false, false_args.as_slice()),
        ],
        _ => Vec::new(),
    }
}

/// The `ClassId`s a terminator passes to its successors, in argument order.
///
/// One flat sequence per terminator: a Jump's args, a Branch's `true_args`
/// followed by its `false_args`, a Ret's value alone. Every pass that indexes
/// terminator arguments -- the pseudo-op builder here, the splitter, and
/// lowering -- numbers them through this one function, so an argument index
/// means the same thing everywhere.
pub(crate) fn terminator_arg_classes(terminator: &EffectfulOp) -> Vec<ClassId> {
    match terminator {
        EffectfulOp::Jump { args, .. } => args.clone(),
        EffectfulOp::Branch {
            true_args,
            false_args,
            ..
        } => true_args.iter().chain(false_args.iter()).copied().collect(),
        EffectfulOp::Ret { val: Some(cid) } => vec![*cid],
        _ => Vec::new(),
    }
}

/// Append the block's terminator arguments to its schedule as a single
/// [`Op::TerminatorArgs`] pseudo-instruction.
///
/// ONE OP CARRYING ALL ARGUMENTS, not one op per argument. Lowering emits an
/// edge's phi copies as a parallel copy (`sequentialize_copies`), which really
/// does need every source readable at one point, so a single op is what the
/// emitted code does. One op per argument instead says argument k dies at its
/// own pseudo-op; the allocator then reuses the registers and the copies read
/// stale values.
///
/// The clique a single op creates is therefore real, and the way to break it is
/// to take an argument out of the register file rather than to lie about
/// liveness: `SplitPlan::operand_removals` drops the operand for an argument
/// that travels through a stack slot instead.
///
/// This must run at the same point as [`populate_effectful_operands`] -- after
/// scheduling, before the splitter -- so that the splitter's operand rewriting
/// and coalescing's renaming both reach these operands. That is the entire
/// point: before this existed the terminator's arguments were `ClassId`s that
/// no rewrite touched, and three separate passes re-derived them from
/// `class_to_vreg`.
///
/// `param_override_vregs` maps a canonical class to the VReg linearization gave
/// this block's own parameter, for the parameters where it minted a fresh one. A
/// back edge passing a parameter straight through resolves to that VReg and to
/// nothing in the snapshot, so it has to be consulted here for the same reason
/// lowering consults it.
pub(super) fn append_terminator_args(
    schedule: &mut Vec<ScheduledInst>,
    terminator: &EffectfulOp,
    block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    param_override_vregs: &BTreeMap<ClassId, VReg>,
    next_vreg: &mut u32,
) -> Result<(), String> {
    let classes = terminator_arg_classes(terminator);
    if classes.is_empty() {
        return Ok(());
    }

    let exit_point = ProgramPoint::block_exit(block_idx);
    let mut arg_indices: Vec<u32> = Vec::with_capacity(classes.len());
    let mut operands: Vec<VReg> = Vec::with_capacity(classes.len());
    for (arg_idx, &cid) in classes.iter().enumerate() {
        let canon = egraph.unionfind.find_immutable(cid);
        // An argument with no VReg at the block exit has nothing for the copy to
        // read, wherever the question is asked. Reporting it here names the
        // argument; skipping it silently is how the old derivations came to
        // disagree about which argument was which.
        let vreg = param_override_vregs
            .get(&canon)
            .copied()
            .or_else(|| class_to_vreg.lookup(canon, exit_point))
            .ok_or_else(|| {
                format!("terminator arg {arg_idx} class {canon:?} has no VReg at block exit")
            })?;
        arg_indices.push(arg_idx as u32);
        operands.push(vreg);
    }

    let dst = VReg(*next_vreg);
    *next_vreg += 1;
    schedule.push(ScheduledInst {
        op: Op::TerminatorArgs(arg_indices),
        dst,
        operands,
    });
    Ok(())
}

/// The terminator arguments a block's schedule carries, as `(arg_idx, vreg)`
/// pairs in argument order.
///
/// Empty for a block whose terminator takes no arguments, and missing an entry
/// for any argument the splitter routed through a stack slot -- so read it by
/// argument index, never by position.
/// Drop the terminator arguments whose operand names one of `vregs`, keeping the
/// argument indices of the survivors.
///
/// A value routed through a stack slot holds no register, so naming it as an
/// operand would ask the allocator for one and put an unwritten register into
/// liveness. Removing the operand is how an argument leaves the register file
/// without anything else having to pretend it was never an argument: it keeps
/// its index, and lowering finds no operand under that index and emits the slot
/// access instead.
pub(crate) fn remove_terminator_arg_operands(
    schedule: &mut [ScheduledInst],
    vregs: &BTreeSet<VReg>,
) {
    for inst in schedule.iter_mut() {
        let Op::TerminatorArgs(arg_indices) = &mut inst.op else {
            continue;
        };
        let keep: Vec<bool> = inst.operands.iter().map(|v| !vregs.contains(v)).collect();
        if keep.iter().all(|&k| k) {
            continue;
        }
        let mut i = 0;
        inst.operands.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        let mut i = 0;
        arg_indices.retain(|_| {
            i += 1;
            keep[i - 1]
        });
    }
}

pub(crate) fn terminator_arg_operands(schedule: &[ScheduledInst]) -> Vec<(u32, VReg)> {
    schedule
        .iter()
        .find_map(|inst| match &inst.op {
            Op::TerminatorArgs(arg_indices) => Some(
                arg_indices
                    .iter()
                    .copied()
                    .zip(inst.operands.iter().copied()),
            ),
            _ => None,
        })
        .map(|pairs| pairs.collect())
        .unwrap_or_default()
}
