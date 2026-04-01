use std::collections::{BTreeMap, HashMap, HashSet};

use crate::egraph::EGraph;
use crate::egraph::extract::VReg;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::{BasicBlock, Function};
use crate::ir::op::{ClassId, Op};
use crate::schedule::scheduler::ScheduledInst;

// ── RPO helpers ───────────────────────────────────────────────────────────────

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

    // Map block id -> block index for fast lookup.
    let id_to_idx: HashMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

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

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Collect all ClassIds that are roots for extraction (used by effectful ops).
fn push_block_class_ids(block: &BasicBlock, out: &mut Vec<ClassId>) {
    for op in &block.ops {
        match op {
            EffectfulOp::Load { addr, result, .. } => {
                out.push(*addr);
                out.push(*result);
            }
            EffectfulOp::Store { addr, val, .. } => {
                out.push(*addr);
                out.push(*val);
            }
            EffectfulOp::Call { args, results, .. } => {
                out.extend_from_slice(args);
                out.extend_from_slice(results);
            }
            EffectfulOp::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                out.push(*cond);
                out.extend_from_slice(true_args);
                out.extend_from_slice(false_args);
            }
            EffectfulOp::Jump { args, .. } => out.extend_from_slice(args),
            EffectfulOp::Ret { val } => {
                if let Some(v) = val {
                    out.push(*v);
                }
            }
        }
    }
}

pub(super) fn collect_roots(func: &Function) -> Vec<ClassId> {
    let mut roots = Vec::new();
    for block in &func.blocks {
        push_block_class_ids(block, &mut roots);
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
            if let EffectfulOp::Call { func: callee, .. } = op {
                if !externals.contains(callee) {
                    externals.push(callee.clone());
                }
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

/// Build a map from (block_id, param_idx) -> canonical ClassId for all block params.
///
/// Scans the egraph for BlockParam nodes and records their canonical class IDs.
pub(super) fn build_block_param_class_map(egraph: &EGraph) -> HashMap<(BlockId, u32), ClassId> {
    let mut map: HashMap<(BlockId, u32), ClassId> = HashMap::new();
    for i in 0..egraph.classes.len() as u32 {
        let cid = ClassId(i);
        let canon = egraph.unionfind.find_immutable(cid);
        if canon != cid {
            continue; // Only process canonical classes.
        }
        let class = egraph.class(cid);
        for node in &class.nodes {
            if let Op::BlockParam(bid, pidx, _) = &node.op {
                map.insert((*bid, *pidx), cid);
            }
        }
    }
    map
}

/// Collect VRegs for all phi-copy source arguments across all blocks.
///
/// These are the values passed as args to Jump/Branch. They need to be in
/// `live_out` so the regalloc doesn't allocate two simultaneously-needed
/// phi source values to the same register (especially on loop back-edges).
pub(super) fn collect_phi_source_vregs(
    func: &Function,
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
    result: &mut HashSet<VReg>,
) {
    for block in &func.blocks {
        for op in &block.ops {
            let args: &[ClassId] = match op {
                EffectfulOp::Jump { args, .. } => args,
                EffectfulOp::Branch {
                    true_args,
                    false_args,
                    ..
                } => {
                    for &cid in true_args.iter().chain(false_args.iter()) {
                        let canon = egraph.unionfind.find_immutable(cid);
                        if let Some(&vreg) = class_to_vreg.get(&canon) {
                            result.insert(vreg);
                        }
                    }
                    continue;
                }
                _ => continue,
            };
            for &cid in args {
                let canon = egraph.unionfind.find_immutable(cid);
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    result.insert(vreg);
                }
            }
        }
    }
}

/// Build phi copy pairs from block parameter passing for coalescing.
///
/// For each Jump/Branch that passes args to a target block with params,
/// for each (arg_class_id, param_class_id) pair, look up their VRegs
/// and add them as copy pairs: (arg_vreg, param_vreg).
pub(super) fn compute_copy_pairs(
    func: &Function,
    class_to_vreg: &HashMap<ClassId, VReg>,
    egraph: &EGraph,
    block_param_map: &HashMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
) -> Vec<(VReg, VReg)> {
    let mut pairs: Vec<(VReg, VReg)> = Vec::new();

    let get_vreg = |cid: ClassId| -> Option<VReg> {
        let canon = egraph.unionfind.find_immutable(cid);
        class_to_vreg.get(&canon).copied()
    };

    // Look up the destination VReg for a block param, preferring the
    // per-block override (fresh VReg) over the global class_to_vreg.
    let get_param_vreg = |target: BlockId, idx: u32, param_cid: ClassId| -> Option<VReg> {
        param_vreg_overrides
            .get(&(target, idx))
            .copied()
            .or_else(|| get_vreg(param_cid))
    };

    for block in &func.blocks {
        for op in &block.ops {
            let (target, args): (BlockId, &[ClassId]) = match op {
                EffectfulOp::Jump { target, args } => (*target, args),
                EffectfulOp::Branch {
                    bb_true, true_args, ..
                } => {
                    // Handle true branch.
                    for (idx, &arg_cid) in true_args.iter().enumerate() {
                        if let Some(&param_cid) = block_param_map.get(&(*bb_true, idx as u32)) {
                            if let (Some(arg_v), Some(param_v)) = (
                                get_vreg(arg_cid),
                                get_param_vreg(*bb_true, idx as u32, param_cid),
                            ) {
                                pairs.push((arg_v, param_v));
                            }
                        }
                    }
                    // Handle false branch via the destructuring below.
                    if let EffectfulOp::Branch {
                        bb_false,
                        false_args,
                        ..
                    } = op
                    {
                        for (idx, &arg_cid) in false_args.iter().enumerate() {
                            if let Some(&param_cid) = block_param_map.get(&(*bb_false, idx as u32))
                            {
                                if let (Some(arg_v), Some(param_v)) = (
                                    get_vreg(arg_cid),
                                    get_param_vreg(*bb_false, idx as u32, param_cid),
                                ) {
                                    pairs.push((arg_v, param_v));
                                }
                            }
                        }
                    }
                    continue;
                }
                _ => continue,
            };
            for (idx, &arg_cid) in args.iter().enumerate() {
                if let Some(&param_cid) = block_param_map.get(&(target, idx as u32)) {
                    if let (Some(arg_v), Some(param_v)) = (
                        get_vreg(arg_cid),
                        get_param_vreg(target, idx as u32, param_cid),
                    ) {
                        pairs.push((arg_v, param_v));
                    }
                }
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
) -> HashMap<VReg, u32> {
    let n = func.blocks.len();
    // Compute per-block loop depth using back-edge counting.
    let mut block_depth: Vec<u32> = vec![0u32; n];

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
                if let Some(target_idx) = func.blocks.iter().position(|b| b.id == target) {
                    if target_idx <= src_idx {
                        // Back-edge: all blocks from target_idx to src_idx are in the loop.
                        for d in block_depth[target_idx..=src_idx].iter_mut() {
                            *d += 1;
                        }
                    }
                }
            }
        }
    }

    // Map each VReg to its block's loop depth.
    let mut result: HashMap<VReg, u32> = HashMap::new();
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
