use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::ir::effectful::{BlockId, EffOperand, EffectfulOp, TermArgs};
use crate::ir::function::BasicBlock;
use crate::ir::op::{ClassId, MachOp, Op, PseudoOp, PureOp};
use crate::ir::types::Type;
use crate::regalloc::{SlotAllocator, SlotOwner};
use crate::schedule::scheduler::ScheduledInst;

/// If the block terminator is a Branch, mark its `cond` VReg as consumed after
/// all non-terminator barriers. This ensures the flags-setting instruction
/// (e.g. X86Sub) is scheduled in the last barrier group, after all calls that
/// would clobber EFLAGS.
pub(super) fn mark_branch_cond_barrier(
    terminator: Option<&EffectfulOp>,
    non_term_count: usize,
    vreg_to_arg: &mut BTreeMap<VReg, usize>,
) {
    if let Some(EffectfulOp::Branch { cond, .. }) = terminator {
        if let Some(vreg) = cond.vreg() {
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
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    schedule: &[ScheduledInst],
) -> (BTreeMap<VReg, usize>, BTreeMap<VReg, usize>) {
    let non_term_count = block.non_term_count();
    let non_term_ops = &block.ops[..non_term_count];
    let (result_map, mut arg_map) =
        build_barrier_maps(non_term_ops, egraph, class_to_vreg, schedule);
    mark_branch_cond_barrier(block.ops.last(), non_term_count, &mut arg_map);
    (result_map, arg_map)
}

/// Build barrier maps: which VRegs are produced/consumed by each effectful op.
///
/// Every VReg *this block's schedule defines* for an argument's class is
/// constrained, rather than the one a point lookup answers with. The point is
/// the thing that cannot be asked for here: resolving a class to a VReg needs a
/// program point, the point depends on the instruction order, and the order is
/// what this map exists to constrain. Asking at block entry answered with the
/// VReg some earlier block emitted, so a class the block re-emitted carried no
/// constraint at all -- the scheduler was free to place its definition after the
/// call that reads it, and the emitted code passed whatever the register held on
/// entry. A class with more than one copy in a block gets all of them
/// constrained, which is what the copies being the same value permits.
pub(super) fn build_barrier_maps(
    non_term_ops: &[EffectfulOp],
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    schedule: &[ScheduledInst],
) -> (BTreeMap<VReg, usize>, BTreeMap<VReg, usize>) {
    let mut vreg_to_result: BTreeMap<VReg, usize> = BTreeMap::new();
    let mut vreg_to_arg: BTreeMap<VReg, usize> = BTreeMap::new();

    // Which class each of this block's VRegs carries. A VReg belongs to one
    // class, so the direction inverts without an ambiguity to resolve.
    let defined: BTreeSet<VReg> = schedule.iter().map(|inst| inst.dst).collect();
    let mut vreg_class: BTreeMap<VReg, ClassId> = BTreeMap::new();
    for (class, vreg) in class_to_vreg.iter() {
        if defined.contains(&vreg) {
            vreg_class.insert(vreg, egraph.unionfind.find_immutable(class));
        }
    }
    let mut class_vregs: BTreeMap<ClassId, Vec<VReg>> = BTreeMap::new();
    for (&vreg, &class) in &vreg_class {
        class_vregs.entry(class).or_default().push(vreg);
    }

    // Mark every VReg of a ClassId as consumed by barrier_k (earliest wins).
    let mark = |target: &mut BTreeMap<VReg, usize>, cid: ClassId, barrier_k: usize| {
        let canon = egraph.unionfind.find_immutable(cid);
        for &vreg in class_vregs.get(&canon).into_iter().flatten() {
            let entry = target.entry(vreg).or_insert(barrier_k);
            *entry = (*entry).min(barrier_k);
        }
    };
    for (barrier_k, op) in non_term_ops.iter().enumerate() {
        match op {
            EffectfulOp::Load { addr, result, .. } => {
                mark(&mut vreg_to_result, result.class(), barrier_k);
                mark(&mut vreg_to_arg, addr.class(), barrier_k);
            }
            EffectfulOp::Store { addr, val, .. } => {
                mark(&mut vreg_to_arg, addr.class(), barrier_k);
                mark(&mut vreg_to_arg, val.class(), barrier_k);
            }
            EffectfulOp::Call { args, results, .. } => {
                for &result_cid in results {
                    mark(&mut vreg_to_result, result_cid.class(), barrier_k);
                }
                for arg in args {
                    mark(&mut vreg_to_arg, arg.class(), barrier_k);
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
        .filter(|i| {
            matches!(
                i.op,
                Op::Mach(MachOp::X86Idiv(..)) | Op::Mach(MachOp::X86Div(..))
            )
        })
        .map(|i| i.dst)
        .collect();
    let div_proj_source = |inst: &ScheduledInst| -> Option<VReg> {
        if !matches!(inst.op, Op::Pure(PureOp::Proj0) | Op::Pure(PureOp::Proj1)) {
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
    slots: &mut SlotAllocator,
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
        let slot = slots.alloc(SlotOwner::EarlyBarrier);

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
            op: Op::Pseudo(PseudoOp::SpillStore(slot as i64)),
            dst: store_vreg,
            operands: vec![*v],
        };
        vreg_group.insert(store_vreg, *def_group);

        let load_inst = ScheduledInst {
            op: Op::Pseudo(PseudoOp::SpillLoad(slot as i64)),
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
pub(crate) fn role_operand_count(op: &EffectfulOp) -> usize {
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
///
/// Nothing here resolves a class. Every VReg this writes -- the roles and the
/// result whose instruction the roles attach to -- is one linearization chose
/// and `cfg::commit_effectful_vregs` wrote into the CFG, so there is no second
/// answer for this to disagree with.
pub(super) fn populate_effectful_operands(
    schedule: &mut Vec<ScheduledInst>,
    non_term_ops: &[EffectfulOp],
    vreg_group: &mut BTreeMap<VReg, usize>,
    next_vreg: &mut u32,
) {
    // Build Addr-child lookup: for each VReg that defines an Addr node,
    // record its operand children.
    let addr_children: BTreeMap<VReg, Vec<VReg>> = schedule
        .iter()
        .filter(|inst| matches!(inst.op, Op::Pure(PureOp::Addr { .. })))
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
        // The folded `Addr` children a role operand needs kept alive, appended
        // after the roles. Roles stay at their own index; only these trailing
        // liveness operands are deduped.
        let append_addr_children = |roles: Vec<VReg>| -> Vec<VReg> {
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

        // The role operands of an op whose operands the CFG states. Read, not
        // resolved: linearization chose the VReg and
        // `cfg::commit_effectful_vregs` wrote it into the CFG, so there
        // is no second answer here to disagree with the first.
        let committed_roles = |ops: &[&EffOperand]| -> Vec<VReg> {
            let mut roles = Vec::with_capacity(ops.len());
            for op in ops {
                let Some(vreg) = op.vreg() else {
                    // Same reason as the resolved path below: dropping a role
                    // shifts every later one down an index, and Phase 7 reads
                    // roles by index.
                    debug_assert!(
                        false,
                        "effectful operand {op:?} has no committed VReg; \
                         skipping it would shift the remaining roles"
                    );
                    continue;
                };
                roles.push(vreg);
            }
            append_addr_children(roles)
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
                let vregs = committed_roles(&[addr]);
                if vregs.is_empty() {
                    continue;
                }
                // The LoadResult instruction is the one defining the result's
                // VReg, which the CFG states.
                let Some(result_vreg) = result.vreg() else {
                    continue;
                };
                if let Some(inst) = schedule.iter_mut().find(|i| i.dst == result_vreg) {
                    append_operands(inst, vregs);
                }
            }
            EffectfulOp::Call { args, results, .. } => {
                let vregs = committed_roles(&args.iter().collect::<Vec<_>>());
                if let Some(first_result) = results.first() {
                    // Non-void call: attach to existing CallResult.
                    if vregs.is_empty() {
                        continue;
                    }
                    let Some(result_vreg) = first_result.vreg() else {
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
                            op: Op::Pseudo(PseudoOp::VoidCallBarrier),
                            dst,
                            operands: vregs,
                        },
                    ));
                }
            }
            EffectfulOp::Store { addr, val, .. } => {
                let vregs = committed_roles(&[addr, val]);
                if vregs.is_empty() {
                    continue;
                }
                let dst = VReg(*next_vreg);
                *next_vreg += 1;
                markers.push((
                    barrier_k,
                    ScheduledInst {
                        op: Op::Pseudo(PseudoOp::StoreBarrier),
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
pub(crate) fn terminator_edges(terminator: &EffectfulOp) -> Vec<(BlockId, &TermArgs)> {
    match terminator {
        EffectfulOp::Jump { target, args } => vec![(*target, args)],
        EffectfulOp::Branch {
            bb_true,
            bb_false,
            true_args,
            false_args,
            ..
        } => vec![(*bb_true, true_args), (*bb_false, false_args)],
        _ => Vec::new(),
    }
}

/// The successor edges a terminator has, mutably.
pub(crate) fn terminator_edges_mut(terminator: &mut EffectfulOp) -> Vec<(BlockId, &mut TermArgs)> {
    match terminator {
        EffectfulOp::Jump { target, args } => vec![(*target, args)],
        EffectfulOp::Branch {
            bb_true,
            bb_false,
            true_args,
            false_args,
            ..
        } => vec![(*bb_true, true_args), (*bb_false, false_args)],
        _ => Vec::new(),
    }
}

/// The `(successor, parameter index)` each terminator argument feeds, in the
/// argument numbering [`append_terminator_args`] defines.
///
/// A `Ret`'s value feeds no parameter, so a `Ret` yields nothing.
pub(crate) fn terminator_arg_destinations(terminator: &EffectfulOp) -> Vec<(BlockId, u32)> {
    terminator_edges(terminator)
        .into_iter()
        .flat_map(|(target, args)| (0..args.len() as u32).map(move |pidx| (target, pidx)))
        .collect()
}

/// The `ClassId` each terminator argument names, in the same numbering as
/// [`terminator_arg_destinations`].
///
/// The class outlives the VReg here: `TermArgs::Committed` stops being
/// maintained at the splitter, but which *expression* an argument is does not
/// change, and that is the only question a store-versus-no-store decision on an
/// edge can be answered by.
pub(crate) fn terminator_arg_classes(terminator: &EffectfulOp) -> Vec<ClassId> {
    terminator_edges(terminator)
        .into_iter()
        .flat_map(|(_, args)| args.class_ids().collect::<Vec<_>>())
        .collect()
}

/// Append the block's terminator arguments to its schedule as a single
/// [`Op::TerminatorArgs`] pseudo-instruction.
///
/// This is where the argument numbering every pass indexes by is defined: one
/// flat sequence per terminator, a Jump's args, then a Branch's `true_args`
/// followed by its `false_args`, or a Ret's value alone.
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
/// A block argument's VReg is not resolved here: it is read off the CFG's
/// [`TermArgs::Committed`], which `cfg::commit_terminator_arg_vregs` wrote once.
/// `BLITZ_DEBUG=phi` traces that choice there, where it is made.
///
/// A `Ret`'s value is the one argument still named by class, so it is the one
/// still resolved -- `Ret.val` stays a `ClassId` because lowering reads it to
/// materialize a constant return straight into the ABI register.
/// `param_override_vregs` is consulted for it and answers on no program
/// measured: it would take a `Ret` of this block's own parameter whose VReg
/// linearization minted fresh, which needs that parameter's class to have been
/// emitted by a non-dominating earlier block.
pub(super) fn append_terminator_args(
    schedule: &mut Vec<ScheduledInst>,
    terminator: &EffectfulOp,
    next_vreg: &mut u32,
) -> Result<(), String> {
    let mut operands: Vec<VReg> = Vec::new();
    match terminator {
        EffectfulOp::Jump { .. } | EffectfulOp::Branch { .. } => {
            for (_, args) in terminator_edges(terminator) {
                let committed = args.as_committed().ok_or_else(|| {
                    "terminator arguments were never committed to VRegs".to_string()
                })?;
                operands.extend(committed.iter().map(|a| a.vreg));
            }
        }
        EffectfulOp::Ret { val: Some(val) } => {
            // A value with no VReg has nothing for the return move to read.
            // Reporting it here names it; skipping it silently leaves the
            // function returning whatever the ABI register happened to hold.
            let vreg = val.vreg().ok_or_else(|| {
                format!(
                    "Ret value class {:?} was never committed to a VReg",
                    val.class()
                )
            })?;
            operands.push(vreg);
        }
        _ => {}
    }
    if operands.is_empty() {
        return Ok(());
    }

    let arg_indices: Vec<u32> = (0..operands.len() as u32).collect();
    let dst = VReg(*next_vreg);
    *next_vreg += 1;
    schedule.push(ScheduledInst {
        op: Op::Pseudo(PseudoOp::TerminatorArgs(arg_indices)),
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
/// Drop the terminator arguments at `arg_indices_to_drop`, keeping the argument
/// indices of the survivors.
///
/// A value routed through a stack slot holds no register, so naming it as an
/// operand would ask the allocator for one and put an unwritten register into
/// liveness. Removing the operand is how an argument leaves the register file
/// without anything else having to pretend it was never an argument: it keeps
/// its index, and lowering finds no operand under that index and emits the slot
/// access instead.
///
/// Keyed on the argument index, never on the operand's VReg. One VReg can fill
/// several argument positions -- an edge whose target has two parameters of the
/// same e-class passes it twice -- and only some of those destinations need be
/// routed. Dropping every operand that names the VReg then silently unhooks a
/// parameter nothing routed, leaving whatever the register happened to hold as
/// the value the target block reads.
pub(crate) fn remove_terminator_arg_operands(
    schedule: &mut [ScheduledInst],
    arg_indices_to_drop: &BTreeSet<u32>,
) {
    for inst in schedule.iter_mut() {
        let Op::Pseudo(PseudoOp::TerminatorArgs(arg_indices)) = &mut inst.op else {
            continue;
        };
        let keep: Vec<bool> = arg_indices
            .iter()
            .map(|idx| !arg_indices_to_drop.contains(idx))
            .collect();
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
            Op::Pseudo(PseudoOp::TerminatorArgs(arg_indices)) => Some(
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
