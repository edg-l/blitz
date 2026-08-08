use std::collections::BTreeMap;

use crate::compile::program_point::ProgramPoint;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::ir::effectful::{BlockId, EffOperand, EffectfulOp, TermArg, TermArgs};
use crate::ir::function::{BasicBlock, Function};
use crate::ir::op::{ClassId, Op, PureOp};
use crate::schedule::scheduler::ScheduledInst;

// ── RPO helpers ───────────────────────────────────────────────────────────────

/// Map each block's `BlockId` to its index in `func.blocks`.
///
/// A terminator names its successors by `BlockId`; the schedules, the liveness
/// sets and every helper here are keyed on the index. This is the one place
/// that translation is written.
pub(crate) fn block_id_to_idx(func: &Function) -> BTreeMap<BlockId, usize> {
    func.blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect()
}

/// The successor `BlockId`s a terminator names, in its own order: a `Jump`'s
/// target, or a `Branch`'s true target then its false target.
///
/// A block with no terminator has no successors.
pub(crate) fn successor_ids(terminator: Option<&EffectfulOp>) -> Vec<BlockId> {
    terminator.map_or_else(Vec::new, |term| {
        super::barrier::terminator_edges(term)
            .into_iter()
            .map(|(target, _)| target)
            .collect()
    })
}

/// The successor block indices of each block, in terminator order.
///
/// A target that names no block in this function is dropped, which is what every
/// caller wants: the CFG passes are index-keyed and cannot represent it.
pub(crate) fn successor_indices(func: &Function) -> Vec<Vec<usize>> {
    let id_to_idx = block_id_to_idx(func);
    func.blocks
        .iter()
        .map(|block| {
            successor_ids(block.ops.last())
                .into_iter()
                .filter_map(|target| id_to_idx.get(&target).copied())
                .collect()
        })
        .collect()
}

/// The predecessor block indices of each block, each list in block order.
///
/// A block reached twice by one predecessor -- both arms of a `Branch` -- appears
/// twice, so a list's length is the block's in-edge count.
pub(crate) fn predecessor_indices(func: &Function) -> Vec<Vec<usize>> {
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); func.blocks.len()];
    for (src_idx, succs) in successor_indices(func).into_iter().enumerate() {
        for succ_idx in succs {
            preds[succ_idx].push(src_idx);
        }
    }
    preds
}

/// Compute a reverse post-order traversal of the CFG starting from block 0.
///
/// Returns a `Vec<usize>` of block *indices* into `func.blocks` (not block IDs)
/// in RPO order. RPO ensures:
///   - Loop headers come before loop bodies.
///   - Fallthrough targets tend to be adjacent, reducing unnecessary jumps.
pub(super) fn compute_rpo(func: &Function) -> Vec<usize> {
    if func.blocks.is_empty() {
        return vec![];
    }

    let n = func.blocks.len();
    let successors = successor_indices(func);

    // Iterative DFS post-order, then reverse.
    let mut post_order: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    // Stack holds (block_index, child_iterator_index).
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;

    while let Some((node, child_idx)) = stack.last_mut() {
        let node = *node;
        if *child_idx < successors[node].len() {
            let next_child = successors[node][*child_idx];
            *child_idx += 1;
            if !visited[next_child] {
                visited[next_child] = true;
                stack.push((next_child, 0));
            }
        } else {
            post_order.push(node);
            stack.pop();
        }
    }

    post_order.reverse();
    let mut rpo = post_order;

    // Any blocks not reachable from block 0 go at the end, in index order, after
    // the reverse so that the entry stays first: `compute_idom` roots the
    // dominator tree at `rpo[0]` while `DomOrder` roots its numbering at block 0,
    // and the two must be the same block.
    for (i, &was_visited) in visited.iter().enumerate() {
        if !was_visited {
            rpo.push(i);
        }
    }

    rpo
}

/// Compute the immediate dominator for each block using the Cooper-Harvey-Kennedy
/// algorithm on RPO order.
pub(super) fn compute_idom(func: &Function, rpo: &[usize]) -> Vec<Option<usize>> {
    let n = func.blocks.len();
    if n == 0 {
        return vec![];
    }

    let preds = predecessor_indices(func);

    let mut rpo_idx = vec![0usize; n];
    for (pos, &block) in rpo.iter().enumerate() {
        rpo_idx[block] = pos;
    }

    let mut idom: Vec<Option<usize>> = vec![None; n];
    let entry = rpo[0];
    idom[entry] = Some(entry);

    let intersect = |mut a: usize, mut b: usize, idom: &[Option<usize>]| -> usize {
        while a != b {
            while rpo_idx[a] > rpo_idx[b] {
                a = idom[a].unwrap();
            }
            while rpo_idx[b] > rpo_idx[a] {
                b = idom[b].unwrap();
            }
        }
        a
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            let mut new_idom: Option<usize> = None;
            for &p in &preds[b] {
                if idom[p].is_some() {
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect(cur, p, &idom),
                    });
                }
            }
            if new_idom != idom[b] {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    idom[entry] = None;
    idom
}

/// Check if block `a` dominates block `b` using the idom tree.
/// Dominance as two integer comparisons, for the callers that ask it a great
/// many times.
///
/// [`dominates`] walks `b`'s immediate-dominator chain, which is fine once and
/// quadratic when a pass asks it for every (block, value) pair -- linearization
/// does, and on a 3763-block function that walk was most of the compile. A
/// depth-first numbering of the dominator tree answers the same question in
/// constant time: `a` dominates `b` exactly when `b`'s entry number lies inside
/// `a`'s entry/exit interval.
///
/// Blocks the entry cannot reach are in no dominator tree and get an empty
/// interval, so nothing dominates them and they dominate nothing but themselves
/// -- which is what the chain walk answers for them too.
pub(super) struct DomOrder {
    tin: Vec<u32>,
    tout: Vec<u32>,
}

impl DomOrder {
    pub(super) fn new(idom: &[Option<usize>]) -> Self {
        let n = idom.len();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (b, &parent) in idom.iter().enumerate() {
            if let Some(p) = parent
                && p != b
            {
                children[p].push(b);
            }
        }
        let mut tin = vec![0u32; n];
        let mut tout = vec![0u32; n];
        let mut clock = 1u32;
        // Explicit stack: a dominator tree can be as deep as the function has
        // blocks, and recursion would overflow on the shapes this exists for.
        let mut stack: Vec<(usize, usize)> = Vec::new();
        if n > 0 {
            tin[0] = clock;
            clock += 1;
            stack.push((0, 0));
        }
        while let Some((node, next)) = stack.pop() {
            if next < children[node].len() {
                stack.push((node, next + 1));
                let child = children[node][next];
                tin[child] = clock;
                clock += 1;
                stack.push((child, 0));
            } else {
                tout[node] = clock;
                clock += 1;
            }
        }
        DomOrder { tin, tout }
    }

    pub(super) fn dominates(&self, a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }
        if self.tin[a] == 0 || self.tin[b] == 0 {
            return false;
        }
        self.tin[a] < self.tin[b] && self.tout[b] < self.tout[a]
    }
}

pub(super) fn dominates(a: usize, b: usize, idom: &[Option<usize>]) -> bool {
    if a == b {
        return true;
    }
    let mut cur = b;
    while let Some(parent) = idom[cur] {
        if parent == a {
            return true;
        }
        cur = parent;
    }
    false
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Collect all ClassIds that are roots for extraction (used by effectful ops).
fn push_block_class_ids(block: &BasicBlock, out: &mut Vec<ClassId>) {
    for op in &block.ops {
        op.for_each_class_id(|c| out.push(c));
    }
}

pub(super) fn collect_roots(func: &Function, egraph: &EGraph) -> Vec<ClassId> {
    let mut roots = Vec::new();
    for block in &func.blocks {
        push_block_class_ids(block, &mut roots);
    }
    for r in &mut roots {
        *r = egraph.unionfind.find_immutable(*r);
    }
    roots.sort_by_key(|c| c.0);
    roots.dedup();
    roots
}

/// Collect external symbol names referenced by Call ops.
pub(super) fn collect_externals(func: &Function) -> Vec<String> {
    let mut externals = Vec::new();
    for block in &func.blocks {
        for op in &block.ops {
            if let EffectfulOp::Call { func: callee, .. } = op
                && !externals.contains(callee)
            {
                externals.push(callee.clone());
            }
        }
    }
    externals
}

// ── Multi-block helpers ───────────────────────────────────────────────────────

/// Collect canonical ClassIds referenced by a single block's effectful ops.
pub(super) fn collect_block_roots(block: &BasicBlock, egraph: &EGraph) -> Vec<ClassId> {
    let mut roots = Vec::new();
    push_block_class_ids(block, &mut roots);
    for r in &mut roots {
        *r = egraph.unionfind.find_immutable(*r);
    }
    roots.sort_by_key(|c| c.0);
    roots.dedup();
    roots
}

/// Commit linearization's choice of VReg for every block argument into the CFG.
///
/// Each `Jump`/`Branch` argument list goes from [`TermArgs::Classes`] to
/// [`TermArgs::Committed`]. After this, "which register carries this argument" is a
/// fact the CFG states rather than a question every later pass answers against a
/// position-keyed map -- the seam that produced seven wrong-code bugs.
///
/// Resolved through the *block's own* snapshot, not the function-wide map: a
/// class re-emitted per block has one VReg per block, and the function-wide map
/// holds whichever block restored it last.
///
/// The overrides answer ahead of the snapshot in two cases:
///
/// - a parameter of this block whose VReg linearization minted fresh, which the
///   snapshot predates -- it names the non-dominating earlier block's VReg for
///   that class instead;
/// - a back edge whose argument *is* the target's parameter. That emits a
///   self-copy: the parameter is the value's storage for the whole loop and the
///   latch has no VReg of its own for it.
///
/// **Neither changes an answer at this point, measured**: over the lit suite and
/// 90 generated programs at both levels, the override and the snapshot never
/// disagree. They existed because the resolution used to happen after the
/// splitter, where the snapshot no longer covers the point -- a header that
/// spills its parameter truncates the class's segment, the value looks dead over
/// the loop body, and the header re-spills a register the latch has since reused.
/// Resolving before the splitter, where every segment is still full-range, is
/// what made them inert. They stay because which answer is right where the two
/// *would* differ is unproven, and the shape is unreached rather than
/// unreachable; step 4 of `docs/internal/refactor-roadmap.md` is where they go.
///
/// Must run after extraction and DCE2, and before the splitter -- the splitter's
/// operand rewriting is what makes a pressure decision stick, and it works on
/// the schedules, which are built from what this writes.
#[allow(clippy::too_many_arguments)]
pub(super) fn commit_terminator_arg_vregs(
    func: &mut Function,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_snapshots: &[ClassVRegMap],
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    rpo_pos: &[usize],
) -> Result<(), String> {
    let id_to_idx = block_id_to_idx(func);
    let func_blocks = &func.blocks;

    let overrides: Vec<BTreeMap<ClassId, VReg>> = func_blocks
        .iter()
        .enumerate()
        .map(|(block_idx, block)| {
            let mut overrides: BTreeMap<ClassId, VReg> = (0..block.param_types.len() as u32)
                .filter_map(|pidx| {
                    let vreg = block.param_vreg(pidx)?;
                    let cid = block_param_map.get(&(block.id, pidx))?;
                    Some((egraph.unionfind.find_immutable(*cid), vreg))
                })
                .collect();
            let Some(term) = block.ops.last() else {
                return overrides;
            };
            for (target, args) in super::barrier::terminator_edges(term) {
                let Some(&target_idx) = id_to_idx.get(&target) else {
                    continue;
                };
                if rpo_pos[block_idx] < rpo_pos[target_idx] {
                    continue; // Forward edge: the argument's own VReg is right.
                }
                for (pidx, &arg_cid) in args.expect_classes().iter().enumerate() {
                    let Some(&param_cid) = block_param_map.get(&(target, pidx as u32)) else {
                        continue;
                    };
                    let canon_param = egraph.unionfind.find_immutable(param_cid);
                    if egraph.unionfind.find_immutable(arg_cid) != canon_param {
                        continue;
                    }
                    let param_vreg =
                        func_blocks[target_idx].param_vreg(pidx as u32).or_else(|| {
                            class_to_vreg.lookup(canon_param, ProgramPoint::block_entry(target_idx))
                        });
                    if let Some(v) = param_vreg {
                        overrides.insert(canon_param, v);
                    }
                }
            }
            overrides
        })
        .collect();

    for (block_idx, block) in func.blocks.iter_mut().enumerate() {
        let snapshot = &block_snapshots[block_idx];
        let exit_point = ProgramPoint::block_exit(block_idx);
        let overrides = &overrides[block_idx];
        let Some(term) = block.ops.last_mut() else {
            continue;
        };
        for (_, args) in super::barrier::terminator_edges_mut(term) {
            let mut committed: Vec<TermArg> = Vec::with_capacity(args.len());
            for (arg_idx, &cid) in args.expect_classes().iter().enumerate() {
                let canon = egraph.unionfind.find_immutable(cid);
                // An argument with no VReg at the block exit has nothing for the
                // copy to read, wherever the question is asked. Reporting it here
                // names the argument; skipping it silently is how the old
                // derivations came to disagree about which argument was which.
                let vreg = overrides
                    .get(&canon)
                    .copied()
                    .or_else(|| snapshot.lookup(canon, exit_point))
                    .ok_or_else(|| {
                        format!(
                            "block {block_idx}: argument {arg_idx} class {canon:?} \
                             has no VReg at block exit"
                        )
                    })?;
                if crate::trace::is_enabled("phi") {
                    tracing::debug!(
                        target: "blitz::phi",
                        "[b{block_idx}] argument {arg_idx} {canon:?} -> {vreg:?} via {}",
                        if overrides.contains_key(&canon) {
                            "OVERRIDE"
                        } else {
                            "snapshot"
                        },
                    );
                }
                committed.push(TermArg { class: canon, vreg });
            }
            *args = TermArgs::Committed(committed);
        }
        // A `Ret`'s value is not a block argument, but it is resolved with the
        // same two answers in the same order: a parameter of this block whose
        // VReg linearization minted fresh is not in the snapshot, which predates
        // it. The blocks that end in a `Ret` have no successors, so the back-edge
        // half of `overrides` is empty for them and this sees the fresh-parameter
        // half alone.
        if let EffectfulOp::Ret { val: Some(val) } = term {
            let canon = egraph.unionfind.find_immutable(val.class());
            let vreg = overrides
                .get(&canon)
                .copied()
                .or_else(|| snapshot.lookup(canon, exit_point))
                .ok_or_else(|| {
                    format!(
                        "block {block_idx}: Ret value class {canon:?} has no VReg at block exit"
                    )
                })?;
            *val = EffOperand::Committed { class: canon, vreg };
        }
    }
    Ok(())
}

/// Commit linearization's choice of VReg for everything a non-terminator
/// effectful op reads or defines, and for a `Branch`'s condition.
///
/// A `Ret`'s value is committed by [`commit_terminator_arg_vregs`] instead,
/// which already holds the parameter overrides its resolution needs.
///
/// The sibling of [`commit_terminator_arg_vregs`], and the same move for the
/// same reason: the address a `Load` reads is a register in a block, while the
/// CFG named it by `ClassId`, which is an expression and has no position.
/// Committing here leaves `populate_effectful_operands` and
/// `add_call_precolors_for_block` copying the answer instead of each resolving
/// it for itself -- which they did through different maps at different points.
///
/// The results are here too, and they are *defs*: the VReg a `Load` writes is
/// the `dst` of the barrier instruction linearization emitted for it, which is
/// the same VReg its class resolves to in that block. Committing it is what lets
/// a consumer find the barrier instruction without asking the map which VReg
/// carries the result.
///
/// Resolved through the *block's own* snapshot, since a class re-emitted per
/// block has one VReg per block and the function-wide map holds whichever block
/// restored it last. Every segment linearization makes is full-range, so the
/// point inside the block cannot distinguish two answers -- the block is what
/// selects, and the point is asked for only because `lookup` takes one.
///
/// An operand with no VReg is an error naming the op, not a skip: the barrier's
/// role operands are positional, so an operand silently dropped there takes a
/// `Store`'s value for its address.
pub(super) fn commit_effectful_vregs(
    func: &mut Function,
    egraph: &EGraph,
    block_snapshots: &[ClassVRegMap],
) -> Result<(), String> {
    for (block_idx, block) in func.blocks.iter_mut().enumerate() {
        let snapshot = &block_snapshots[block_idx];
        let point = ProgramPoint::block_exit(block_idx);
        for (op_idx, op) in block.ops.iter_mut().enumerate() {
            let roles: Vec<(String, &mut EffOperand)> = match op {
                EffectfulOp::Load { addr, result, .. } => vec![
                    ("Load address".to_string(), addr),
                    ("Load result".to_string(), result),
                ],
                EffectfulOp::Store { addr, val, .. } => {
                    vec![
                        ("Store address".to_string(), addr),
                        ("Store value".to_string(), val),
                    ]
                }
                EffectfulOp::Call { args, results, .. } => args
                    .iter_mut()
                    .enumerate()
                    .map(|(i, arg)| (format!("Call argument {i}"), arg))
                    .chain(
                        results
                            .iter_mut()
                            .enumerate()
                            .map(|(i, r)| (format!("Call result {i}"), r)),
                    )
                    .collect(),
                EffectfulOp::Branch { cond, .. } => {
                    vec![("Branch condition".to_string(), cond)]
                }
                _ => continue,
            };
            for (what, operand) in roles {
                let canon = egraph.unionfind.find_immutable(operand.class());
                let vreg = snapshot.lookup(canon, point).ok_or_else(|| {
                    format!(
                        "block {block_idx}, effectful op {op_idx}: {what} class {canon:?} \
                         has no VReg in that block"
                    )
                })?;
                *operand = EffOperand::Committed { class: canon, vreg };
            }
        }
    }
    Ok(())
}

/// Commit linearization's choice of VReg for every block parameter into the CFG.
///
/// The destination end of the phi seam, as [`commit_terminator_arg_vregs`] is
/// the argument end. Linearization decides this in one place and it was recorded
/// in a side map threaded through six signatures; the block is where it belongs.
pub(super) fn commit_block_param_vregs(
    func: &mut Function,
    block_param_vregs: &BTreeMap<(BlockId, u32), VReg>,
) {
    for block in func.blocks.iter_mut() {
        block.param_vregs = (0..block.param_types.len() as u32)
            .map(|pidx| block_param_vregs.get(&(block.id, pidx)).copied())
            .collect();
    }
}

/// The VReg block `target`'s parameter `pidx` is written into, from the three
/// places that can name it, in the order that matters.
///
/// Every pass that touches a phi copy has to agree on this answer: the copy that
/// writes the parameter, the coalescer deciding which VRegs may share a register,
/// and the allocator's parameter sets. Two passes deriving it separately is how
/// coalescing came to merge a parameter onto a VReg the copy never wrote.
///
/// 1. **The target block's own `Op::BlockParam`.** That is the VReg the block
///    reads, so a copy into anything else writes a register nobody looks at. One
///    class can name several VRegs and here the two answers came apart: the class
///    resolved to a VReg in RAX while the block's schedule read RSI, so a loop
///    counter started at whatever RSI held and the loop was skipped entirely. It
///    answers 122223 of 123595 resolutions measured, and is the only source that
///    survives coalescing: past that point the other two name the pre-merge VReg.
/// 2. **The class map at the target's entry**, because where a reload covers that
///    point the reload is what the block reads. It answers 442 of the 1372 where
///    the schedule does not, and differs from 3 in 2 of them.
/// 3. **The VReg the CFG states**, `BasicBlock::param_vregs`. A parameter passing
///    a dominating definition straight through gets no `BlockParam` of its own,
///    so once the splitter cross-block-spills that definition and truncates its
///    segment to the defining block, nothing else names the value here -- 9 of 40
///    generated programs failed to compile at -O1 on exactly that.
///
/// A fourth source was here and is gone: the override linearization minted for a
/// parameter whose class an earlier non-dominating block emitted. It answered
/// **0 of 123595** resolutions, because linearization gives such a parameter a
/// `BlockParam` instruction of its own and source 1 then answers first.
pub(super) fn resolve_block_param_vreg(
    target_block: &BasicBlock,
    pidx: u32,
    target_idx: usize,
    target_schedule: &[ScheduledInst],
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
) -> Option<VReg> {
    let target = target_block.id;
    let from_schedule = target_schedule.iter().find_map(|inst| match inst.op {
        Op::Pure(PureOp::BlockParam(bid, i, _)) if bid == target && i == pidx => Some(inst.dst),
        _ => None,
    });
    let from_map = || -> Option<VReg> {
        let param_cid = *block_param_map.get(&(target, pidx))?;
        let canon = egraph.unionfind.find_immutable(param_cid);
        class_to_vreg.lookup(canon, ProgramPoint::block_entry(target_idx))
    };
    let from_cfg = target_block.param_vreg(pidx);
    if crate::trace::is_enabled("paramsrc") {
        let answer = from_schedule.or_else(from_map).or(from_cfg);
        let disagree = [from_schedule, from_map(), from_cfg]
            .iter()
            .any(|v| v.is_some() && *v != answer);
        if disagree {
            tracing::debug!(
                target: "blitz::paramsrc",
                "b{target}.p{pidx} -> {:?}: schedule={:?} map={:?} cfg={:?}",
                answer.map(|v| v.0),
                from_schedule.map(|v| v.0),
                from_map().map(|v| v.0),
                from_cfg.map(|v| v.0),
            );
        }
    }
    from_schedule.or_else(from_map).or(from_cfg)
}

/// Build phi copy pairs from the schedules, as `(arg_vreg, param_vreg)`.
///
/// The same pairs [`compute_copy_pairs`] derives from `class_to_vreg`, except
/// that each argument's VReg is the operand its block's `Op::TerminatorArgs`
/// carries and each parameter's VReg is the one the target block's own
/// `Op::BlockParam` defines. Those are the VRegs the emitted copy reads and
/// writes, so they are the pairs coalescing may merge; a class resolved through
/// the function-wide map can instead name a VReg defined in a block that does
/// not reach this edge, and merging *that* forces the parameter into a register
/// chosen for an unrelated value.
///
/// An argument with no operand travels through a stack slot and needs no copy,
/// so it contributes no pair.
pub(super) fn compute_copy_pairs_from_schedules(
    func: &Function,
    block_schedules: &[Vec<ScheduledInst>],
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
) -> Vec<(VReg, VReg)> {
    let id_to_idx = block_id_to_idx(func);

    // Resolved exactly as the copy that writes the parameter resolves it, so
    // coalescing merges onto the VReg that copy targets and not another.
    let param_vreg = |target: BlockId, pidx: u32| -> Option<VReg> {
        let target_idx = *id_to_idx.get(&target)?;
        resolve_block_param_vreg(
            func.blocks.get(target_idx)?,
            pidx,
            target_idx,
            block_schedules.get(target_idx)?,
            egraph,
            class_to_vreg,
            block_param_map,
        )
    };

    let mut pairs: Vec<(VReg, VReg)> = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let Some(term) = block.ops.last() else {
            continue;
        };
        let dests = super::barrier::terminator_arg_destinations(term);
        let Some(schedule) = block_schedules.get(block_idx) else {
            continue;
        };
        for (arg_idx, arg_v) in super::barrier::terminator_arg_operands(schedule) {
            let Some(&(target, pidx)) = dests.get(arg_idx as usize) else {
                continue;
            };
            if let Some(param_v) = param_vreg(target, pidx) {
                pairs.push((arg_v, param_v));
            }
        }
    }
    pairs
}

/// The copy pairs a projection makes, one per `Proj0` that lowering will emit a
/// `mov` for.
///
/// **A pair-producing x86 op keeps its result in the pair VReg and `Proj0` reads
/// it out**, so `lower.rs` emits `mov proj, pair` whenever the two got different
/// registers -- and nothing offered that pair to coalescing, because
/// `compute_copy_pairs_from_schedules` collects phi copies and only those. On
/// `bench` this is the single largest group of surviving copies: an `X86Add`
/// lowered to `lea ebx, [rdi+r10]` followed by `mov r12d, ebx` and then a use of
/// r12d, where the `lea` could have written r12d in the first place.
///
/// Which projections qualify is `interference::result_shares_operand`, the same
/// predicate the interference graph excludes its edge from -- a division is not
/// one, because `X86Idiv`/`X86Div` leave their results in RAX and RDX and the
/// pair VReg holds nothing.
pub(super) fn projection_copy_pairs(block_schedules: &[Vec<ScheduledInst>]) -> Vec<(VReg, VReg)> {
    let mut pairs: Vec<(VReg, VReg)> = Vec::new();
    for schedule in block_schedules {
        let div_dsts = super::lower::division_dst_vregs(schedule);
        for inst in schedule {
            if !matches!(inst.op, Op::Pure(PureOp::Proj0)) {
                continue;
            }
            // The same predicate the interference graph draws its edges from,
            // so a pair offered here is never one the graph forbids.
            let Some(k) = crate::regalloc::interference::result_shares_operand(inst, &div_dsts)
            else {
                continue;
            };
            let Some(&pair) = inst.operands.get(k) else {
                continue;
            };
            pairs.push((pair, inst.dst));
        }
    }
    pairs
}

/// Loop depth per block index, by back-edge counting.
///
/// A target at or before its source is a back edge, and every block between the
/// two is in the loop. A heuristic rather than a dominator-tree loop forest,
/// which is what the callers weight costs by: how many times a block's code is
/// expected to run, relative to the rest.
pub(super) fn block_loop_depths(func: &Function) -> Vec<u32> {
    let mut depth = vec![0u32; func.blocks.len()];
    for (src_idx, succs) in successor_indices(func).into_iter().enumerate() {
        for target_idx in succs {
            if target_idx <= src_idx {
                for d in depth[target_idx..=src_idx].iter_mut() {
                    *d += 1;
                }
            }
        }
    }
    depth
}

/// Compute loop depth for each VReg based on the CFG back-edges.
///
/// A back-edge is a jump/branch to a block with a lower (or equal) index,
/// indicating a loop. All VRegs defined in blocks within the loop body get
/// a non-zero depth. This is a simple heuristic (not a full dominator tree).
pub(super) fn compute_loop_depths(
    func: &Function,
    block_schedules: &[Vec<ScheduledInst>],
) -> BTreeMap<VReg, u32> {
    let block_depth = block_loop_depths(func);

    // Map each VReg to its block's loop depth.
    let mut result: BTreeMap<VReg, u32> = BTreeMap::new();
    for (block_idx, sched) in block_schedules.iter().enumerate() {
        let depth = block_depth[block_idx];
        if depth == 0 {
            continue;
        }
        for inst in sched {
            result.insert(inst.dst, depth);
        }
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// The block-parameter VRegs the allocator must treat as written at block entry.
///
/// Three sources, and the order matters. `collect_block_param_vregs_per_block`
/// finds a parameter only where a `ClassVRegMap` segment still covers the block
/// entry; `lower_terminator` falls back to the block's own `param_vregs`, so a
/// parameter the splitter truncated is a parameter there and not one here, and
/// the allocator would draw no interference edge to its siblings -- leaving
/// coalescing free to merge two parameters of one block into one register, so
/// both phi copies target it and the second overwrites the first.
///
/// Slot-routed parameters then come back out. They have no register at all:
/// predecessors store them and uses load them, so listing one would make the
/// allocator treat it as live-in, extend its range to every predecessor's exit
/// and produce reloads for a value already in a slot.
pub(super) fn collect_alloc_block_params(
    func: &Function,
    egraph: &EGraph,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    class_to_vreg: &ClassVRegMap,
    slot_spilled_params: &super::split::BlockParamSlotMap,
    block_id_to_idx: &BTreeMap<BlockId, usize>,
) -> Vec<crate::regalloc::interference::VRegSet> {
    let mut out = crate::regalloc::global_liveness::collect_block_param_vregs_per_block(
        func,
        egraph,
        block_param_map,
        class_to_vreg,
    );

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (pidx, vreg) in block.param_vregs.iter().enumerate() {
            let Some(&vreg) = vreg.as_ref() else { continue };
            if slot_spilled_params.contains_key(&(block.id, pidx as u32)) {
                continue;
            }
            out[block_idx].insert(vreg.0 as usize);
        }
    }

    for (&(bid, pidx), info) in slot_spilled_params {
        let block_idx = block_id_to_idx[&bid];
        out[block_idx].remove(info.vreg.0 as usize);
        if let Some(own) = func.blocks[block_idx].param_vreg(pidx) {
            out[block_idx].remove(own.0 as usize);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::enode::ENode;
    use crate::ir::{CondCode, Type};

    /// An e-graph of `n` distinct I64 constants, class `i` holding `iconst(i)`,
    /// so a `ClassId` in a test is its own canonical representative.
    fn constants(n: u32) -> EGraph {
        let mut egraph = EGraph::new();
        for i in 0..n {
            let cid = egraph.add(ENode {
                op: Op::Pure(PureOp::Iconst(i as i64, Type::I64)),
                children: Default::default(),
            });
            assert_eq!(cid, ClassId(i));
        }
        egraph
    }

    /// A class map whose every entry covers the whole function.
    fn map_of(entries: impl IntoIterator<Item = (u32, u32)>) -> ClassVRegMap {
        let mut map = ClassVRegMap::new();
        for (class, vreg) in entries {
            map.insert_full_range(ClassId(class), VReg(vreg));
        }
        map
    }

    fn rpo_positions(func: &Function) -> Vec<usize> {
        let mut pos = vec![0usize; func.blocks.len()];
        for (i, &idx) in compute_rpo(func).iter().enumerate() {
            pos[idx] = i;
        }
        pos
    }

    /// A function whose blocks are wired by the terminators given, block `i`
    /// carrying `terms[i]`. Ids equal indices, so a `BlockId` and an index are
    /// interchangeable in the assertions below.
    fn func_with(terms: Vec<EffectfulOp>) -> Function {
        let blocks = terms
            .into_iter()
            .enumerate()
            .map(|(i, t)| BasicBlock {
                id: i as BlockId,
                param_types: vec![],
                param_vregs: vec![],
                ops: vec![t],
            })
            .collect::<Vec<_>>();
        Function {
            name: "f".to_string(),
            param_types: vec![],
            return_types: vec![],
            next_block_id: blocks.len() as BlockId,
            blocks,
            param_class_ids: vec![],
            egraph: None,
            stack_slots: vec![],
            noinline: false,
        }
    }

    fn jump(target: BlockId) -> EffectfulOp {
        EffectfulOp::Jump {
            target,
            args: TermArgs::classes([]),
        }
    }

    fn branch(bb_true: BlockId, bb_false: BlockId) -> EffectfulOp {
        EffectfulOp::Branch {
            cond: EffOperand::Class(ClassId(0)),
            cc: CondCode::Ne,
            bb_true,
            bb_false,
            true_args: TermArgs::classes([]),
            false_args: TermArgs::classes([]),
        }
    }

    fn ret() -> EffectfulOp {
        EffectfulOp::Ret { val: None }
    }

    /// 0 -> {1, 2} -> 3.
    fn diamond() -> Function {
        func_with(vec![branch(1, 2), jump(3), jump(3), ret()])
    }

    /// 0 -> 1, 1 -> {2, 3}, 2 -> 1 (back edge), 3 returns.
    fn loop_with_exit() -> Function {
        func_with(vec![jump(1), branch(2, 3), jump(1), ret()])
    }

    /// 0 returns; 1 and 2 form a cycle the entry cannot reach.
    fn unreachable_cycle() -> Function {
        func_with(vec![ret(), jump(2), jump(1)])
    }

    fn idom_of(func: &Function) -> Vec<Option<usize>> {
        compute_idom(func, &compute_rpo(func))
    }

    /// The whole point of `DomOrder`: it is a constant-time restatement of the
    /// chain walk, so the two must answer identically for every ordered pair of
    /// blocks. `DomOrder` numbers from block 0 and the chain walk roots at
    /// `rpo[0]`, so a function with unreachable blocks is the shape that pulls
    /// them apart and belongs in the loop.
    #[test]
    fn dom_order_agrees_with_the_chain_walk() {
        for func in [diamond(), loop_with_exit(), unreachable_cycle()] {
            let idom = idom_of(&func);
            let order = DomOrder::new(&idom);
            for a in 0..func.blocks.len() {
                for b in 0..func.blocks.len() {
                    assert_eq!(
                        order.dominates(a, b),
                        dominates(a, b, &idom),
                        "{}: disagreement on ({a}, {b}), idom = {idom:?}",
                        func.name,
                    );
                }
            }
        }
    }

    #[test]
    fn entry_dominates_every_reachable_block_and_neither_arm_dominates_the_join() {
        let func = diamond();
        let idom = idom_of(&func);
        let order = DomOrder::new(&idom);
        for b in 0..func.blocks.len() {
            assert!(order.dominates(0, b), "entry must dominate block {b}");
        }
        assert!(!order.dominates(1, 3));
        assert!(!order.dominates(2, 3));
        assert!(!order.dominates(1, 2));
        assert!(!order.dominates(3, 0));
    }

    #[test]
    fn a_loop_header_dominates_its_body_and_its_exit() {
        let func = loop_with_exit();
        let order = DomOrder::new(&idom_of(&func));
        assert!(order.dominates(1, 2), "the header must dominate the body");
        assert!(order.dominates(1, 3), "the header must dominate the exit");
        assert!(
            !order.dominates(2, 1),
            "the body must not dominate the header"
        );
    }

    /// A block outside the dominator tree gets an empty interval, so it
    /// dominates nothing but itself and nothing dominates it.
    #[test]
    fn nothing_dominates_an_unreachable_block() {
        let func = unreachable_cycle();
        let order = DomOrder::new(&idom_of(&func));
        for b in [1, 2] {
            assert!(!order.dominates(0, b));
            assert!(!order.dominates(b, 0));
            assert!(order.dominates(b, b));
        }
        assert!(!order.dominates(1, 2));
    }

    /// RPO's stated guarantee: a loop header comes before its body, and every
    /// block appears exactly once even when the entry cannot reach it.
    #[test]
    fn rpo_puts_a_loop_header_before_its_body_and_lists_every_block_once() {
        let func = loop_with_exit();
        let rpo = compute_rpo(&func);
        let mut seen = rpo.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..func.blocks.len()).collect::<Vec<_>>());
        let pos = |b: usize| rpo.iter().position(|&x| x == b).unwrap();
        assert!(pos(1) < pos(2), "header before body, got {rpo:?}");
        assert!(pos(0) < pos(1));

        let func = unreachable_cycle();
        let rpo = compute_rpo(&func);
        assert_eq!(rpo[0], 0, "the entry comes first, got {rpo:?}");
        let mut seen = rpo;
        seen.sort_unstable();
        assert_eq!(
            seen,
            vec![0, 1, 2],
            "unreachable blocks are still listed once"
        );
    }

    #[test]
    fn successors_and_predecessors_are_the_same_edge_set() {
        for func in [diamond(), loop_with_exit(), unreachable_cycle()] {
            let succs = successor_indices(&func);
            let preds = predecessor_indices(&func);
            let mut from_succs: Vec<(usize, usize)> = succs
                .iter()
                .enumerate()
                .flat_map(|(a, ss)| ss.iter().map(move |&b| (a, b)))
                .collect();
            let mut from_preds: Vec<(usize, usize)> = preds
                .iter()
                .enumerate()
                .flat_map(|(b, ps)| ps.iter().map(move |&a| (a, b)))
                .collect();
            from_succs.sort_unstable();
            from_preds.sort_unstable();
            assert_eq!(from_succs, from_preds, "{}", func.name);
        }
    }

    // ── The phi seam: which VReg carries a parameter, and which an argument ──

    /// A block with `n` parameters, whose recorded VRegs are `vregs`.
    fn block_with_params(id: BlockId, vregs: Vec<Option<VReg>>, term: EffectfulOp) -> BasicBlock {
        BasicBlock {
            id,
            param_types: vec![Type::I64; vregs.len()],
            param_vregs: vregs,
            ops: vec![term],
        }
    }

    fn block_param_inst(target: BlockId, pidx: u32, dst: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::BlockParam(target, pidx, Type::I64)),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    /// The three sources answer in order, and each takes over exactly where the
    /// one before it has nothing to say.
    #[test]
    fn a_block_parameters_vreg_comes_from_the_schedule_then_the_map_then_the_cfg() {
        let egraph = constants(1);
        let param_map = BTreeMap::from([((0 as BlockId, 0u32), ClassId(0))]);
        let schedule = vec![block_param_inst(0, 0, 10)];
        let map = map_of([(0, 20)]);
        let block = block_with_params(0, vec![Some(VReg(30))], ret());

        let resolve = |block: &BasicBlock, schedule: &[ScheduledInst], map: &ClassVRegMap| {
            resolve_block_param_vreg(block, 0, 0, schedule, &egraph, map, &param_map)
        };

        assert_eq!(
            resolve(&block, &schedule, &map),
            Some(VReg(10)),
            "the block's own BlockParam is the VReg its schedule reads"
        );
        assert_eq!(
            resolve(&block, &[], &map),
            Some(VReg(20)),
            "with no BlockParam, the class map at block entry answers"
        );
        assert_eq!(
            resolve(&block, &[], &ClassVRegMap::new()),
            Some(VReg(30)),
            "with neither, the VReg the CFG states answers"
        );
        let unnamed = block_with_params(0, vec![None], ret());
        assert_eq!(resolve(&unnamed, &[], &ClassVRegMap::new()), None);
    }

    /// A `BlockParam` for another block, or another parameter of this one, is not
    /// this parameter's answer -- the schedule holds one per parameter.
    #[test]
    fn only_this_blocks_own_block_param_answers_for_it() {
        let egraph = constants(1);
        let param_map = BTreeMap::new();
        let schedule = vec![block_param_inst(1, 0, 10), block_param_inst(0, 1, 11)];
        let block = block_with_params(0, vec![None, None], ret());
        assert_eq!(
            resolve_block_param_vreg(
                &block,
                0,
                0,
                &schedule,
                &egraph,
                &ClassVRegMap::new(),
                &param_map,
            ),
            None,
        );
    }

    #[test]
    fn committing_block_param_vregs_states_what_linearization_chose() {
        let mut func = func_with(vec![jump(1), ret()]);
        func.blocks[1].param_types = vec![Type::I64; 2];
        commit_block_param_vregs(
            &mut func,
            &BTreeMap::from([((1 as BlockId, 1u32), VReg(7))]),
        );
        assert_eq!(func.blocks[0].param_vregs, Vec::<Option<VReg>>::new());
        assert_eq!(func.blocks[1].param_vregs, vec![None, Some(VReg(7))]);
    }

    /// Two blocks passing the same class to the same parameter commit their own
    /// block's VReg, not whichever the function-wide map holds.
    #[test]
    fn a_terminator_argument_commits_the_vreg_of_its_own_block() {
        let egraph = constants(1);
        let mut func = func_with(vec![branch(1, 2), jump(3), jump(3), ret()]);
        for idx in [1, 2] {
            let EffectfulOp::Jump { args, .. } = func.blocks[idx].ops.last_mut().unwrap() else {
                unreachable!()
            };
            *args = TermArgs::classes([ClassId(0)]);
        }
        func.blocks[3].param_types = vec![Type::I64];

        let snapshots = vec![
            ClassVRegMap::new(),
            map_of([(0, 7)]),
            map_of([(0, 8)]),
            ClassVRegMap::new(),
        ];
        let rpo_pos = rpo_positions(&func);
        commit_terminator_arg_vregs(
            &mut func,
            &egraph,
            &map_of([(0, 99)]),
            &snapshots,
            &BTreeMap::from([((3 as BlockId, 0u32), ClassId(0))]),
            &rpo_pos,
        )
        .unwrap();

        for (idx, vreg) in [(1, VReg(7)), (2, VReg(8))] {
            let EffectfulOp::Jump { args, .. } = func.blocks[idx].ops.last().unwrap() else {
                unreachable!()
            };
            assert_eq!(
                args.as_committed(),
                Some(
                    [TermArg {
                        class: ClassId(0),
                        vreg
                    }]
                    .as_slice()
                ),
                "block {idx}",
            );
        }
    }

    /// A back edge whose argument *is* the target's parameter emits a self-copy:
    /// the parameter is the value's storage for the whole loop, so the latch
    /// commits the parameter's VReg rather than one of its own.
    #[test]
    fn a_back_edge_argument_that_is_the_parameter_commits_the_parameters_vreg() {
        let egraph = constants(1);
        let mut func = func_with(vec![jump(1), branch(2, 3), jump(1), ret()]);
        func.blocks[1] = block_with_params(1, vec![Some(VReg(5))], branch(2, 3));
        let EffectfulOp::Jump { args, .. } = func.blocks[2].ops.last_mut().unwrap() else {
            unreachable!()
        };
        *args = TermArgs::classes([ClassId(0)]);

        let snapshots = vec![
            ClassVRegMap::new(),
            ClassVRegMap::new(),
            map_of([(0, 42)]),
            ClassVRegMap::new(),
        ];
        let rpo_pos = rpo_positions(&func);
        commit_terminator_arg_vregs(
            &mut func,
            &egraph,
            &ClassVRegMap::new(),
            &snapshots,
            &BTreeMap::from([((1 as BlockId, 0u32), ClassId(0))]),
            &rpo_pos,
        )
        .unwrap();

        let EffectfulOp::Jump { args, .. } = func.blocks[2].ops.last().unwrap() else {
            unreachable!()
        };
        assert_eq!(
            args.as_committed().unwrap()[0].vreg,
            VReg(5),
            "the latch must write the parameter the header reads, not its own VReg",
        );
    }

    /// An argument with no VReg at the block exit is an error naming it. Skipping
    /// it silently is how the old derivations came to disagree about which
    /// argument was which.
    #[test]
    fn an_argument_with_no_vreg_is_an_error_naming_the_argument() {
        let egraph = constants(2);
        let mut func = func_with(vec![jump(1), ret()]);
        let EffectfulOp::Jump { args, .. } = func.blocks[0].ops.last_mut().unwrap() else {
            unreachable!()
        };
        *args = TermArgs::classes([ClassId(0), ClassId(1)]);
        func.blocks[1].param_types = vec![Type::I64; 2];

        let snapshots = vec![map_of([(0, 7)]), ClassVRegMap::new()];
        let rpo_pos = rpo_positions(&func);
        let err = commit_terminator_arg_vregs(
            &mut func,
            &egraph,
            &ClassVRegMap::new(),
            &snapshots,
            &BTreeMap::new(),
            &rpo_pos,
        )
        .unwrap_err();
        assert!(err.contains("argument 1"), "{err}");
        assert!(err.contains("ClassId(1)"), "{err}");
    }

    /// A `Ret`'s value is not a block argument, but the same commit resolves it
    /// through the same snapshot.
    #[test]
    fn a_ret_value_is_committed_through_the_returning_blocks_snapshot() {
        let egraph = constants(1);
        let mut func = func_with(vec![EffectfulOp::Ret {
            val: Some(EffOperand::Class(ClassId(0))),
        }]);
        let rpo_pos = rpo_positions(&func);
        commit_terminator_arg_vregs(
            &mut func,
            &egraph,
            &ClassVRegMap::new(),
            &[map_of([(0, 4)])],
            &BTreeMap::new(),
            &rpo_pos,
        )
        .unwrap();
        let EffectfulOp::Ret { val: Some(val) } = func.blocks[0].ops.last().unwrap() else {
            unreachable!()
        };
        assert_eq!(
            *val,
            EffOperand::Committed {
                class: ClassId(0),
                vreg: VReg(4),
            }
        );
    }
}
