use std::collections::{BTreeMap, BTreeSet};

use crate::compile::program_point::ProgramPoint;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::ir::effectful::{BlockId, EffectfulOp, TermArg, TermArgs};
use crate::ir::function::{BasicBlock, Function};
use crate::ir::op::{ClassId, Op};
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

    // Build a successor map: block index -> list of successor block indices.
    let n = func.blocks.len();

    let id_to_idx = block_id_to_idx(func);

    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|block| {
            let mut succs = Vec::new();
            if let Some(term) = block.ops.last() {
                match term {
                    EffectfulOp::Jump { target, .. } => {
                        if let Some(&idx) = id_to_idx.get(target) {
                            succs.push(idx);
                        }
                    }
                    EffectfulOp::Branch {
                        bb_true, bb_false, ..
                    } => {
                        if let Some(&idx) = id_to_idx.get(bb_true) {
                            succs.push(idx);
                        }
                        if let Some(&idx) = id_to_idx.get(bb_false) {
                            succs.push(idx);
                        }
                    }
                    _ => {}
                }
            }
            succs
        })
        .collect();

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

    // Any blocks not reachable from block 0 are appended at the end in index order.
    for (i, &was_visited) in visited.iter().enumerate() {
        if !was_visited {
            post_order.push(i);
        }
    }

    post_order.reverse();
    post_order
}

/// Compute the immediate dominator for each block using the Cooper-Harvey-Kennedy
/// algorithm on RPO order.
pub(super) fn compute_idom(func: &Function, rpo: &[usize]) -> Vec<Option<usize>> {
    let n = func.blocks.len();
    if n == 0 {
        return vec![];
    }

    let id_to_idx = block_id_to_idx(func);
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, block) in func.blocks.iter().enumerate() {
        if let Some(term) = block.ops.last() {
            let succs: Vec<usize> = match term {
                EffectfulOp::Jump { target, .. } => {
                    id_to_idx.get(target).copied().into_iter().collect()
                }
                EffectfulOp::Branch {
                    bb_true, bb_false, ..
                } => {
                    let mut v = Vec::new();
                    if let Some(&idx) = id_to_idx.get(bb_true) {
                        v.push(idx);
                    }
                    if let Some(&idx) = id_to_idx.get(bb_false) {
                        v.push(idx);
                    }
                    v
                }
                _ => vec![],
            };
            for s in succs {
                preds[s].push(i);
            }
        }
    }

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

/// Collect VRegs for all phi-copy source arguments across all blocks.
///
/// These are the values passed as args to Jump/Branch. They need to be in
/// `live_out` so the regalloc doesn't allocate two simultaneously-needed
/// phi source values to the same register (especially on loop back-edges).
///
/// Read straight off the CFG, which names them by VReg once
/// [`commit_terminator_arg_vregs`] has run.
pub(super) fn collect_phi_source_vregs(func: &Function, result: &mut BTreeSet<VReg>) {
    for block in &func.blocks {
        let Some(term) = block.ops.last() else {
            continue;
        };
        for (_, args) in super::barrier::terminator_edges(term) {
            result.extend(
                args.as_committed()
                    .unwrap_or_default()
                    .iter()
                    .map(|a| a.vreg),
            );
        }
    }
}

/// Build phi copy pairs from block parameter passing for coalescing.
///
/// One pair per argument the CFG names and destination parameter that resolves,
/// as `(arg_vreg, param_vreg)`. Used on the single-block path only; the
/// multi-block path takes its pairs from the schedules, after the splitter, via
/// [`compute_copy_pairs_from_schedules`].
pub(super) fn compute_copy_pairs(
    func: &Function,
    class_to_vreg: &ClassVRegMap,
    egraph: &EGraph,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
) -> Vec<(VReg, VReg)> {
    let mut pairs: Vec<(VReg, VReg)> = Vec::new();

    let id_to_idx = block_id_to_idx(func);

    for block in &func.blocks {
        let Some(term) = block.ops.last() else {
            continue;
        };
        for (target, args) in super::barrier::terminator_edges(term) {
            let Some(&target_idx) = id_to_idx.get(&target) else {
                continue;
            };
            let entry_point = ProgramPoint::block_entry(target_idx);
            for (idx, arg) in args.as_committed().unwrap_or_default().iter().enumerate() {
                // The destination VReg for a block param, preferring the
                // per-block override (fresh VReg) over the global class map.
                let param_v = param_vreg_overrides
                    .get(&(target, idx as u32))
                    .copied()
                    .or_else(|| {
                        let param_cid = *block_param_map.get(&(target, idx as u32))?;
                        let canon = egraph.unionfind.find_immutable(param_cid);
                        class_to_vreg.lookup(canon, entry_point)
                    });
                if let Some(param_v) = param_v {
                    pairs.push((arg.vreg, param_v));
                }
            }
        }
    }
    pairs
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
/// unreachable; step 4 of `docs/refactor-roadmap.md` is where they go.
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
    block_param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    rpo_pos: &[usize],
) -> Result<(), String> {
    let id_to_idx = block_id_to_idx(func);

    let overrides: Vec<BTreeMap<ClassId, VReg>> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(block_idx, block)| {
            let mut overrides: BTreeMap<ClassId, VReg> = block_param_vreg_overrides
                .iter()
                .filter(|((bid, _), _)| *bid == block.id)
                .filter_map(|(&(bid, pidx), &fresh)| {
                    let cid = block_param_map.get(&(bid, pidx))?;
                    Some((egraph.unionfind.find_immutable(*cid), fresh))
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
                    let param_vreg = block_param_vreg_overrides
                        .get(&(target, pidx as u32))
                        .copied()
                        .or_else(|| {
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
    }
    Ok(())
}

/// The VReg block `target`'s parameter `pidx` is written into, from the four
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
///    counter started at whatever RSI held and the loop was skipped entirely.
/// 2. **The override linearization minted**, for a parameter whose class was
///    already emitted by a non-dominating earlier block.
/// 3. **The class map at the target's entry**, because where a reload covers that
///    point the reload is what the block reads.
/// 4. **What linearization recorded.** A parameter passing a dominating definition
///    straight through gets no `BlockParam` of its own, so once the splitter
///    cross-block-spills that definition and truncates its segment to the defining
///    block, nothing else names the value here -- 9 of 40 generated programs
///    failed to compile at -O1 on exactly that.
pub(super) fn resolve_block_param_vreg(
    target: BlockId,
    pidx: u32,
    target_idx: usize,
    target_schedule: &[ScheduledInst],
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    block_param_vregs: &BTreeMap<(BlockId, u32), VReg>,
) -> Option<VReg> {
    target_schedule
        .iter()
        .find_map(|inst| match inst.op {
            Op::BlockParam(bid, i, _) if bid == target && i == pidx => Some(inst.dst),
            _ => None,
        })
        .or_else(|| param_vreg_overrides.get(&(target, pidx)).copied())
        .or_else(|| {
            let param_cid = *block_param_map.get(&(target, pidx))?;
            let canon = egraph.unionfind.find_immutable(param_cid);
            class_to_vreg.lookup(canon, ProgramPoint::block_entry(target_idx))
        })
        .or_else(|| block_param_vregs.get(&(target, pidx)).copied())
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
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    block_param_vregs: &BTreeMap<(BlockId, u32), VReg>,
) -> Vec<(VReg, VReg)> {
    let id_to_idx = block_id_to_idx(func);

    // Resolved exactly as the copy that writes the parameter resolves it, so
    // coalescing merges onto the VReg that copy targets and not another.
    let param_vreg = |target: BlockId, pidx: u32| -> Option<VReg> {
        let target_idx = *id_to_idx.get(&target)?;
        resolve_block_param_vreg(
            target,
            pidx,
            target_idx,
            block_schedules.get(target_idx)?,
            egraph,
            class_to_vreg,
            block_param_map,
            param_vreg_overrides,
            block_param_vregs,
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

/// Compute loop depth for each VReg based on the CFG back-edges.
///
/// A back-edge is a jump/branch to a block with a lower (or equal) index,
/// indicating a loop. All VRegs defined in blocks within the loop body get
/// a non-zero depth. This is a simple heuristic (not a full dominator tree).
pub(super) fn compute_loop_depths(
    func: &Function,
    block_schedules: &[Vec<ScheduledInst>],
) -> BTreeMap<VReg, u32> {
    let n = func.blocks.len();
    // Compute per-block loop depth using back-edge counting.
    let mut block_depth: Vec<u32> = vec![0u32; n];

    let id_to_idx = block_id_to_idx(func);

    // For each block, check its terminator for back-edges.
    for (src_idx, block) in func.blocks.iter().enumerate() {
        if let Some(terminator) = block.ops.last() {
            let targets: Vec<BlockId> = match terminator {
                EffectfulOp::Jump { target, .. } => vec![*target],
                EffectfulOp::Branch {
                    bb_true, bb_false, ..
                } => vec![*bb_true, *bb_false],
                _ => vec![],
            };
            for target in targets {
                // Find target block index.
                if let Some(&target_idx) = id_to_idx.get(&target)
                    && target_idx <= src_idx
                {
                    // Back-edge: all blocks from target_idx to src_idx are in the loop.
                    for d in block_depth[target_idx..=src_idx].iter_mut() {
                        *d += 1;
                    }
                }
            }
        }
    }

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
