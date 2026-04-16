use std::collections::{BTreeMap, BTreeSet, HashMap};

use smallvec::smallvec;

use crate::egraph::{EGraph, ENode};
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::{BasicBlock, Function};
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

use super::cfg::{compute_idom, compute_rpo, dominates};

/// Extra roots to add to specific blocks during linearization.
/// Maps block_index -> Vec<ClassId> of invariant classes to emit there.
pub type ExtraRoots = BTreeMap<usize, Vec<ClassId>>;

/// Information about a single natural loop detected in the CFG.
pub(super) struct LoopInfo {
    pub header_idx: usize,
    pub body: BTreeSet<usize>,
}

/// Build a predecessor map and a BlockId -> block index map for the function.
///
/// Returns `(preds, id_to_idx)` where `preds[i]` is the list of predecessor
/// block indices for block `i`, and `id_to_idx` maps `BlockId` to block index.
pub(super) fn build_predecessor_map(
    func: &Function,
) -> (Vec<Vec<usize>>, BTreeMap<BlockId, usize>) {
    let n = func.blocks.len();
    let id_to_idx: BTreeMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (src_idx, block) in func.blocks.iter().enumerate() {
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
            for succ in succs {
                preds[succ].push(src_idx);
            }
        }
    }

    (preds, id_to_idx)
}

/// Detect back edges in the CFG using the dominator tree.
///
/// A back edge is an edge `(src, tgt)` where `tgt` dominates `src`.
/// Returns a list of `(src_idx, tgt_idx)` pairs.
pub(super) fn detect_back_edges(
    func: &Function,
    rpo: &[usize],
    idom: &[Option<usize>],
) -> Vec<(usize, usize)> {
    let id_to_idx: BTreeMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    let mut back_edges = Vec::new();

    for (src_idx, block) in func.blocks.iter().enumerate() {
        if let Some(term) = block.ops.last() {
            let targets: Vec<usize> = match term {
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
            for tgt_idx in targets {
                // A back edge exists when the target dominates (or is) the source.
                // Self-loops where src == tgt are included via dominates(a, a) == true.
                if dominates(tgt_idx, src_idx, idom) {
                    back_edges.push((src_idx, tgt_idx));
                }
            }
        }
    }

    // Sort for determinism, following RPO positions.
    let _ = rpo; // rpo was used to compute idom; sorting by (src, tgt) is already deterministic.
    back_edges.sort();
    back_edges
}

/// Collect the body of a natural loop given its header and a back-edge source.
///
/// Performs a backward predecessor walk from `back_edge_src` up to `header_idx`,
/// returning all block indices reachable this way (including both endpoints).
pub(super) fn collect_loop_body(
    header_idx: usize,
    back_edge_src: usize,
    preds: &[Vec<usize>],
) -> BTreeSet<usize> {
    let mut body = BTreeSet::new();
    body.insert(header_idx);

    // Worklist: blocks to process whose predecessors need to be added.
    let mut worklist: Vec<usize> = Vec::new();
    if body.insert(back_edge_src) {
        worklist.push(back_edge_src);
    }

    while let Some(block) = worklist.pop() {
        for &pred in &preds[block] {
            if body.insert(pred) && pred != header_idx {
                worklist.push(pred);
            }
        }
    }

    body
}

/// Detect all natural loops in the function's CFG.
///
/// Groups back edges by their header block and unions the corresponding loop
/// bodies. Returns loops sorted by header RPO position (outermost first).
pub(super) fn detect_loops(func: &Function) -> Vec<LoopInfo> {
    if func.blocks.is_empty() {
        return vec![];
    }

    let rpo = compute_rpo(func);
    let idom = compute_idom(func, &rpo);
    let (preds, _) = build_predecessor_map(func);
    let back_edges = detect_back_edges(func, &rpo, &idom);

    if back_edges.is_empty() {
        return vec![];
    }

    // Group back edges by header (target of the back edge).
    let mut header_to_back_edges: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for edge @ (_, tgt) in &back_edges {
        header_to_back_edges.entry(*tgt).or_default().push(*edge);
    }

    // Build RPO position map for sorting.
    let mut rpo_pos = vec![0usize; func.blocks.len()];
    for (pos, &idx) in rpo.iter().enumerate() {
        rpo_pos[idx] = pos;
    }

    // Build a LoopInfo for each header.
    let mut loops: Vec<LoopInfo> = header_to_back_edges
        .into_iter()
        .map(|(header_idx, edges)| {
            let mut body = BTreeSet::new();
            for &(src, _tgt) in &edges {
                let partial = collect_loop_body(header_idx, src, &preds);
                body.extend(partial);
            }
            LoopInfo { header_idx, body }
        })
        .collect();

    // Sort by RPO position of the header so outermost loops come first.
    loops.sort_by_key(|l| rpo_pos[l.header_idx]);
    loops
}

/// Redirect a predecessor block's terminator from `old_target_id` to `new_target_id`.
///
/// For each block index in `pred_indices`, any `Jump` or `Branch` edge that
/// currently points to `old_target_id` is updated to point to `new_target_id`.
fn redirect_predecessors(
    func: &mut Function,
    old_target_id: BlockId,
    new_target_id: BlockId,
    pred_indices: &[usize],
) {
    for &pred_idx in pred_indices {
        if let Some(term) = func.blocks[pred_idx].ops.last_mut() {
            match term {
                EffectfulOp::Jump { target, .. } => {
                    if *target == old_target_id {
                        *target = new_target_id;
                    }
                }
                EffectfulOp::Branch {
                    bb_true, bb_false, ..
                } => {
                    if *bb_true == old_target_id {
                        *bb_true = new_target_id;
                    }
                    if *bb_false == old_target_id {
                        *bb_false = new_target_id;
                    }
                }
                _ => {}
            }
        }
    }
}

/// Insert a preheader block for the given loop and return its index in `func.blocks`.
///
/// The preheader:
/// - Gets fresh block parameters matching the loop header's parameters.
/// - Has a single `Jump` terminator forwarding those parameters to the header.
/// - Receives all non-back-edge predecessors of the header (predecessors outside
///   the loop body).
///
/// If the header is the entry block (index 0) there are no outside predecessors to
/// redirect, so no preheader is created and `loop_info.header_idx` is returned.
pub(super) fn insert_preheader(
    func: &mut Function,
    egraph: &mut EGraph,
    loop_info: &LoopInfo,
    _id_to_idx: &BTreeMap<BlockId, usize>,
) -> usize {
    let header_idx = loop_info.header_idx;

    // Entry block as header: nothing to redirect.
    if header_idx == 0 {
        return header_idx;
    }

    let header_id = func.blocks[header_idx].id;
    let header_param_types: Vec<Type> = func.blocks[header_idx].param_types.clone();

    // Allocate a fresh BlockId for the preheader.
    let preheader_id = func.fresh_block_id();

    // Build BlockParam e-nodes for each parameter, collecting their ClassIds.
    let param_class_ids: Vec<ClassId> = header_param_types
        .iter()
        .enumerate()
        .map(|(param_idx, ty)| {
            egraph.add(ENode {
                op: Op::BlockParam(preheader_id, param_idx as u32, ty.clone()),
                children: smallvec![],
            })
        })
        .collect();

    // The preheader jumps unconditionally to the header, forwarding its params.
    let preheader_term = EffectfulOp::Jump {
        target: header_id,
        args: param_class_ids,
    };

    let mut preheader = BasicBlock::new(preheader_id, header_param_types);
    preheader.ops.push(preheader_term);

    // Determine which predecessors of the header are NOT back-edge sources
    // (i.e. they lie outside the loop body).
    let non_back_edge_preds: Vec<usize> = {
        let (preds, _) = build_predecessor_map(func);
        preds[header_idx]
            .iter()
            .copied()
            .filter(|pred_idx| !loop_info.body.contains(pred_idx))
            .collect()
    };

    // Redirect those predecessors to point at the preheader instead of the header.
    redirect_predecessors(func, header_id, preheader_id, &non_back_edge_preds);

    // Append the preheader and return its index.
    let preheader_idx = func.blocks.len();
    func.blocks.push(preheader);
    preheader_idx
}

/// Collect all ClassIds that are "defined" (produced) inside the loop body.
///
/// A class is loop-defined if it is a result/output of an effectful op in a
/// loop-body block (Load result, Call results) or a BlockParam of a loop-body
/// block. Operands that are merely *used* by the loop (addr, val, cond, args)
/// are NOT included; those may be loop-invariant.
pub(super) fn collect_loop_defined_classes(
    func: &Function,
    egraph: &EGraph,
    loop_body: &BTreeSet<usize>,
) -> BTreeSet<ClassId> {
    let mut defined: BTreeSet<ClassId> = BTreeSet::new();

    // Collect ClassIds *produced* by effectful ops in loop-body blocks.
    for &block_idx in loop_body {
        let block = &func.blocks[block_idx];
        for op in &block.ops {
            match op {
                EffectfulOp::Load { result, .. } => {
                    defined.insert(egraph.unionfind.find_immutable(*result));
                }
                EffectfulOp::Call { results, .. } => {
                    for &r in results {
                        defined.insert(egraph.unionfind.find_immutable(r));
                    }
                }
                // Store, Branch, Jump, Ret produce no new values.
                _ => {}
            }
        }
    }

    // Collect BlockParam ClassIds for loop-body blocks by scanning all egraph classes.
    let loop_block_ids: BTreeSet<BlockId> =
        loop_body.iter().map(|&idx| func.blocks[idx].id).collect();

    for i in 0..egraph.classes.len() as u32 {
        let cid = ClassId(i);
        let canon = egraph.unionfind.find_immutable(cid);
        if canon != cid {
            continue; // Only process canonical classes.
        }
        let class = egraph.class(cid);
        for node in &class.nodes {
            if let Op::BlockParam(bid, _, _) = &node.op
                && loop_block_ids.contains(bid)
            {
                defined.insert(canon);
            }
        }
    }

    defined
}

/// Check whether a class is loop-invariant.
///
/// A class is loop-invariant if:
///   (a) It is NOT in `loop_defined`, and
///   (b) At least one node in the class has a non-effectful op and all of its
///       children are also recursively loop-invariant.
///
/// Results are cached in `cache` to avoid redundant work. `class_id` is
/// canonicalized before any check.
pub(super) fn is_class_loop_invariant(
    class_id: ClassId,
    egraph: &EGraph,
    loop_defined: &BTreeSet<ClassId>,
    cache: &mut HashMap<ClassId, bool>,
) -> bool {
    let canon = egraph.unionfind.find_immutable(class_id);

    if let Some(&cached) = cache.get(&canon) {
        return cached;
    }

    // (a) Class defined inside the loop is never invariant.
    if loop_defined.contains(&canon) {
        cache.insert(canon, false);
        return false;
    }

    // (b) Look for at least one node whose op is pure and whose children are
    //     all recursively invariant.
    let class = egraph.class(canon);
    let result = class.nodes.iter().any(|node| {
        // Effectful placeholder ops are not invariant.
        let is_effectful = matches!(
            node.op,
            Op::LoadResult(..) | Op::CallResult(..) | Op::StoreBarrier | Op::VoidCallBarrier
        );
        if is_effectful {
            return false;
        }
        // All children must be loop-invariant. Skip ClassId::NONE sentinels.
        node.children.iter().all(|&child| {
            if child == ClassId::NONE {
                return true;
            }
            is_class_loop_invariant(child, egraph, loop_defined, cache)
        })
    });

    cache.insert(canon, result);
    result
}

/// Collect all ClassIds referenced by effectful ops in the given blocks.
fn collect_effectful_operands(func: &Function, block_indices: &BTreeSet<usize>) -> Vec<ClassId> {
    let mut out = Vec::new();
    for &block_idx in block_indices {
        let block = &func.blocks[block_idx];
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
    out
}

/// Find all loop-invariant classes reachable from the loop body.
///
/// Walks the e-graph transitively from all ClassIds referenced by effectful ops
/// in loop-body blocks. Returns the invariant classes found along the way; these
/// are the candidates for hoisting into the preheader.
pub(super) fn find_invariant_classes(
    func: &Function,
    egraph: &EGraph,
    loop_info: &LoopInfo,
) -> Vec<ClassId> {
    let loop_defined = collect_loop_defined_classes(func, egraph, &loop_info.body);
    let mut cache: HashMap<ClassId, bool> = HashMap::new();

    // Seed: all ClassIds directly referenced by effectful ops in loop body.
    let seeds = collect_effectful_operands(func, &loop_info.body);

    // Transitive walk: explore e-graph children to find invariant subexpressions.
    let mut visited: BTreeSet<ClassId> = BTreeSet::new();
    let mut worklist: Vec<ClassId> = Vec::new();
    for cid in seeds {
        let canon = egraph.unionfind.find_immutable(cid);
        if visited.insert(canon) {
            worklist.push(canon);
        }
    }

    let mut invariant: Vec<ClassId> = Vec::new();
    while let Some(cid) = worklist.pop() {
        // Check invariance and collect if true.
        if is_class_loop_invariant(cid, egraph, &loop_defined, &mut cache) {
            invariant.push(cid);
        }

        // Walk children of all nodes in this class.
        let class = egraph.class(cid);
        for node in &class.nodes {
            for &child in &node.children {
                if child != ClassId::NONE {
                    let child_canon = egraph.unionfind.find_immutable(child);
                    if visited.insert(child_canon) {
                        worklist.push(child_canon);
                    }
                }
            }
        }
    }

    invariant
}

/// Run LICM: detect loops, insert preheaders, identify invariant classes.
/// Returns extra roots that the linearization phase should include.
pub fn run_licm(func: &mut Function, egraph: &mut EGraph) -> ExtraRoots {
    let loops = detect_loops(func);

    if loops.is_empty() {
        return BTreeMap::new();
    }

    let trace = crate::trace::is_enabled("licm") && crate::trace::fn_matches(&func.name);

    if trace {
        eprintln!("[licm] fn={}: {} loop(s) detected", func.name, loops.len());
    }

    let mut extra_roots: ExtraRoots = BTreeMap::new();
    let mut total_hoisted = 0usize;

    for loop_info in &loops {
        // Rebuild id_to_idx each iteration because insert_preheader may append blocks.
        let (_, id_to_idx) = build_predecessor_map(func);

        let preheader_idx = insert_preheader(func, egraph, loop_info, &id_to_idx);

        // Entry block as header: preheader was not inserted, skip hoisting.
        if preheader_idx == loop_info.header_idx {
            if trace {
                eprintln!(
                    "[licm]   loop header={} (entry block): skipped hoisting",
                    loop_info.header_idx
                );
            }
            continue;
        }

        let invariant_classes = find_invariant_classes(func, egraph, loop_info);

        if trace {
            eprintln!(
                "[licm]   loop header={} body_size={} hoisted={}",
                loop_info.header_idx,
                loop_info.body.len(),
                invariant_classes.len()
            );
        }

        if !invariant_classes.is_empty() {
            total_hoisted += invariant_classes.len();
            extra_roots.insert(preheader_idx, invariant_classes);
        }
    }

    if trace {
        eprintln!(
            "[licm] fn={}: total hoisted classes={}",
            func.name, total_hoisted
        );
    }

    extra_roots
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::function::{BasicBlock, Function};
    use crate::ir::op::ClassId;

    fn make_func() -> Function {
        let mut f = Function::new("test", vec![], vec![]);
        f.next_block_id = 10; // ensure fresh IDs don't clash with manual IDs
        f
    }

    fn jump(target: BlockId) -> EffectfulOp {
        EffectfulOp::Jump {
            target,
            args: vec![],
        }
    }

    fn branch(cond: ClassId, bb_true: BlockId, bb_false: BlockId) -> EffectfulOp {
        EffectfulOp::Branch {
            cond,
            cc: crate::ir::condcode::CondCode::Ne,
            bb_true,
            bb_false,
            true_args: vec![],
            false_args: vec![],
        }
    }

    fn ret() -> EffectfulOp {
        EffectfulOp::Ret { val: None }
    }

    /// Build a simple while loop:
    ///
    /// ```
    /// bb0 -> bb1 (entry to header)
    /// bb1 -> bb2 (true) | bb3 (false)   -- loop header / condition check
    /// bb2 -> bb1                          -- back edge
    /// bb3 -> ret                          -- exit
    /// ```
    fn build_simple_while() -> Function {
        let mut f = make_func();

        // bb0: entry, jumps to header bb1
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(1));
        f.blocks.push(bb0);

        // bb1: loop header, branches to body (bb2) or exit (bb3)
        let mut bb1 = BasicBlock::new(1, vec![]);
        bb1.ops.push(branch(ClassId(0), 2, 3));
        f.blocks.push(bb1);

        // bb2: loop body, jumps back to header
        let mut bb2 = BasicBlock::new(2, vec![]);
        bb2.ops.push(jump(1));
        f.blocks.push(bb2);

        // bb3: exit
        let mut bb3 = BasicBlock::new(3, vec![]);
        bb3.ops.push(ret());
        f.blocks.push(bb3);

        f
    }

    /// Build nested loops:
    ///
    /// ```
    /// bb0 -> bb1 (outer header)
    /// bb1 -> bb2 | bb5           -- outer condition
    /// bb2 -> bb3 (inner header)
    /// bb3 -> bb4 | bb1           -- inner condition; false exits outer
    /// bb4 -> bb3                  -- inner back edge
    /// bb5 -> ret
    /// ```
    fn build_nested_loops() -> Function {
        let mut f = make_func();

        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(1));
        f.blocks.push(bb0);

        // bb1: outer header
        let mut bb1 = BasicBlock::new(1, vec![]);
        bb1.ops.push(branch(ClassId(0), 2, 5));
        f.blocks.push(bb1);

        // bb2: entry to inner loop
        let mut bb2 = BasicBlock::new(2, vec![]);
        bb2.ops.push(jump(3));
        f.blocks.push(bb2);

        // bb3: inner header — true goes to body bb4, false exits outer to bb1
        let mut bb3 = BasicBlock::new(3, vec![]);
        bb3.ops.push(branch(ClassId(0), 4, 1));
        f.blocks.push(bb3);

        // bb4: inner body, back edge to inner header bb3
        let mut bb4 = BasicBlock::new(4, vec![]);
        bb4.ops.push(jump(3));
        f.blocks.push(bb4);

        // bb5: exit
        let mut bb5 = BasicBlock::new(5, vec![]);
        bb5.ops.push(ret());
        f.blocks.push(bb5);

        f
    }

    // ── build_predecessor_map ─────────────────────────────────────────────────

    #[test]
    fn test_predecessor_map_simple_while() {
        let f = build_simple_while();
        let (preds, id_to_idx) = build_predecessor_map(&f);

        // Indices: bb0=0, bb1=1, bb2=2, bb3=3
        assert_eq!(id_to_idx[&0], 0);
        assert_eq!(id_to_idx[&1], 1);
        assert_eq!(id_to_idx[&2], 2);
        assert_eq!(id_to_idx[&3], 3);

        // bb0 has no predecessors
        assert!(preds[0].is_empty());
        // bb1 is preceded by bb0 and bb2 (back edge)
        let mut p1 = preds[1].clone();
        p1.sort();
        assert_eq!(p1, vec![0, 2]);
        // bb2 is preceded by bb1
        assert_eq!(preds[2], vec![1]);
        // bb3 is preceded by bb1
        assert_eq!(preds[3], vec![1]);
    }

    // ── detect_back_edges ─────────────────────────────────────────────────────

    #[test]
    fn test_back_edges_simple_while() {
        let f = build_simple_while();
        let rpo = compute_rpo(&f);
        let idom = compute_idom(&f, &rpo);
        let back_edges = detect_back_edges(&f, &rpo, &idom);

        // Only one back edge: bb2 (idx 2) -> bb1 (idx 1)
        assert_eq!(back_edges, vec![(2, 1)]);
    }

    #[test]
    fn test_back_edges_nested_loops() {
        let f = build_nested_loops();
        let rpo = compute_rpo(&f);
        let idom = compute_idom(&f, &rpo);
        let back_edges = detect_back_edges(&f, &rpo, &idom);

        // Two back edges: bb4->bb3 (inner) and bb3->bb1 (outer)
        // Indices: bb0=0, bb1=1, bb2=2, bb3=3, bb4=4, bb5=5
        assert!(back_edges.contains(&(4, 3)), "inner back edge bb4->bb3");
        assert!(back_edges.contains(&(3, 1)), "outer back edge bb3->bb1");
        assert_eq!(back_edges.len(), 2);
    }

    #[test]
    fn test_no_back_edges_no_loops() {
        // Linear chain: bb0->bb1->bb2->ret
        let mut f = make_func();
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(1));
        f.blocks.push(bb0);
        let mut bb1 = BasicBlock::new(1, vec![]);
        bb1.ops.push(jump(2));
        f.blocks.push(bb1);
        let mut bb2 = BasicBlock::new(2, vec![]);
        bb2.ops.push(ret());
        f.blocks.push(bb2);

        let rpo = compute_rpo(&f);
        let idom = compute_idom(&f, &rpo);
        let back_edges = detect_back_edges(&f, &rpo, &idom);
        assert!(back_edges.is_empty());
    }

    #[test]
    fn test_self_loop_back_edge() {
        // bb0 jumps to itself
        let mut f = make_func();
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(0));
        f.blocks.push(bb0);

        let rpo = compute_rpo(&f);
        let idom = compute_idom(&f, &rpo);
        let back_edges = detect_back_edges(&f, &rpo, &idom);

        // Self-loop: (0, 0)
        assert_eq!(back_edges, vec![(0, 0)]);
    }

    // ── collect_loop_body ─────────────────────────────────────────────────────

    #[test]
    fn test_collect_loop_body_simple_while() {
        let f = build_simple_while();
        let (preds, _) = build_predecessor_map(&f);
        // Back edge: bb2 (idx 2) -> bb1 (idx 1)
        let body = collect_loop_body(1, 2, &preds);
        // Body must contain header (1) and back-edge source (2)
        assert!(body.contains(&1));
        assert!(body.contains(&2));
        // bb0 and bb3 are outside the loop
        assert!(!body.contains(&0));
        assert!(!body.contains(&3));
    }

    #[test]
    fn test_collect_loop_body_self_loop() {
        let mut f = make_func();
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(0));
        f.blocks.push(bb0);

        let (preds, _) = build_predecessor_map(&f);
        let body = collect_loop_body(0, 0, &preds);
        assert_eq!(body, BTreeSet::from([0]));
    }

    // ── detect_loops ─────────────────────────────────────────────────────────

    #[test]
    fn test_detect_loops_no_loops() {
        let mut f = make_func();
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(ret());
        f.blocks.push(bb0);

        let loops = detect_loops(&f);
        assert!(loops.is_empty());
    }

    #[test]
    fn test_detect_loops_simple_while() {
        let f = build_simple_while();
        let loops = detect_loops(&f);

        assert_eq!(loops.len(), 1);
        let lp = &loops[0];
        // Header is bb1 (index 1)
        assert_eq!(lp.header_idx, 1);
        // Body contains header (1) and body block (2)
        assert!(lp.body.contains(&1));
        assert!(lp.body.contains(&2));
        assert!(!lp.body.contains(&0));
        assert!(!lp.body.contains(&3));
        assert_eq!(lp.body.len(), 2);
    }

    #[test]
    fn test_detect_loops_nested() {
        let f = build_nested_loops();
        let loops = detect_loops(&f);

        assert_eq!(loops.len(), 2, "expected 2 loops (outer + inner)");

        // Outer loop header is bb1 (idx 1), inner is bb3 (idx 3).
        let outer = loops
            .iter()
            .find(|l| l.header_idx == 1)
            .expect("outer loop");
        let inner = loops
            .iter()
            .find(|l| l.header_idx == 3)
            .expect("inner loop");

        // Outer body contains bb1..bb4 but not bb0 or bb5.
        assert!(outer.body.contains(&1));
        assert!(outer.body.contains(&2));
        assert!(outer.body.contains(&3));
        assert!(outer.body.contains(&4));
        assert!(!outer.body.contains(&0));
        assert!(!outer.body.contains(&5));

        // Inner body contains bb3 and bb4.
        assert!(inner.body.contains(&3));
        assert!(inner.body.contains(&4));
        assert!(!inner.body.contains(&1));

        // Outer comes first in RPO order.
        assert!(loops[0].header_idx == 1, "outer loop should be first");
    }

    #[test]
    fn test_detect_loops_self_loop() {
        let mut f = make_func();
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(0));
        f.blocks.push(bb0);

        let loops = detect_loops(&f);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header_idx, 0);
        assert_eq!(loops[0].body, BTreeSet::from([0]));
    }

    // ── insert_preheader ──────────────────────────────────────────────────────

    /// Helper: build id_to_idx map from a function.
    fn id_to_idx(func: &Function) -> BTreeMap<BlockId, usize> {
        func.blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect()
    }

    /// Preheader is inserted between bb0 and the loop header bb1.
    /// After insertion, bb0 should jump to the preheader, bb2 (back-edge) still
    /// jumps to bb1, and the preheader jumps to bb1.
    #[test]
    fn test_insert_preheader_simple_while() {
        let mut f = build_simple_while();
        // Set next_block_id past the existing IDs (0-3).
        f.next_block_id = 10;
        let mut egraph = crate::egraph::EGraph::new();

        let loops = detect_loops(&f);
        assert_eq!(loops.len(), 1);
        let loop_info = &loops[0];

        let map = id_to_idx(&f);
        let preheader_idx = insert_preheader(&mut f, &mut egraph, loop_info, &map);

        // A new block should have been appended.
        assert_eq!(preheader_idx, 4, "preheader is the 5th block (index 4)");
        assert_eq!(f.blocks.len(), 5);

        let preheader = &f.blocks[preheader_idx];
        let header_id = f.blocks[loop_info.header_idx].id; // BlockId 1

        // Preheader's terminator must be a Jump to the header.
        match &preheader.ops[0] {
            EffectfulOp::Jump { target, .. } => {
                assert_eq!(*target, header_id, "preheader must jump to header");
            }
            other => panic!("expected Jump, got {:?}", other),
        }

        // bb0 (index 0) must now jump to the preheader, not to bb1.
        match &f.blocks[0].ops[0] {
            EffectfulOp::Jump { target, .. } => {
                assert_eq!(
                    *target, preheader.id,
                    "bb0 must be redirected to the preheader"
                );
            }
            other => panic!("expected Jump in bb0, got {:?}", other),
        }

        // bb2 (index 2, back-edge source) must still jump to bb1 (the header).
        match &f.blocks[2].ops[0] {
            EffectfulOp::Jump { target, .. } => {
                assert_eq!(*target, header_id, "back-edge bb2 must still target header");
            }
            other => panic!("expected Jump in bb2, got {:?}", other),
        }
    }

    /// Preheader param_types must mirror the loop header's param_types.
    #[test]
    fn test_preheader_param_types_match_header() {
        use crate::ir::types::Type;

        let mut f = make_func();
        f.next_block_id = 10;

        // bb0: entry, jumps to header with no args
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(1));
        f.blocks.push(bb0);

        // bb1: loop header with two parameters
        let mut bb1 = BasicBlock::new(1, vec![Type::I64, Type::I32]);
        bb1.ops.push(branch(ClassId(0), 2, 3));
        f.blocks.push(bb1);

        // bb2: loop body, jumps back with placeholder args
        let mut bb2 = BasicBlock::new(2, vec![]);
        bb2.ops.push(EffectfulOp::Jump {
            target: 1,
            args: vec![ClassId(1), ClassId(2)],
        });
        f.blocks.push(bb2);

        // bb3: exit
        let mut bb3 = BasicBlock::new(3, vec![]);
        bb3.ops.push(ret());
        f.blocks.push(bb3);

        let mut egraph = crate::egraph::EGraph::new();
        let loops = detect_loops(&f);
        assert_eq!(loops.len(), 1);
        let map = id_to_idx(&f);
        let preheader_idx = insert_preheader(&mut f, &mut egraph, &loops[0], &map);

        let header_params = f.blocks[loops[0].header_idx].param_types.clone();
        let preheader_params = f.blocks[preheader_idx].param_types.clone();
        assert_eq!(
            preheader_params, header_params,
            "preheader must have the same param_types as the header"
        );
    }

    /// When the loop header is the entry block (index 0), insert_preheader must
    /// return header_idx unchanged and not modify the function.
    #[test]
    fn test_insert_preheader_entry_block_is_header() {
        let mut f = make_func();
        f.next_block_id = 10;

        // Self-loop on entry block.
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(0));
        f.blocks.push(bb0);

        let mut egraph = crate::egraph::EGraph::new();
        let loops = detect_loops(&f);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header_idx, 0);

        let map = id_to_idx(&f);
        let result_idx = insert_preheader(&mut f, &mut egraph, &loops[0], &map);

        // No preheader should be inserted; function unchanged.
        assert_eq!(
            result_idx, 0,
            "should return header_idx for entry-block header"
        );
        assert_eq!(f.blocks.len(), 1, "no new block should be added");
    }

    /// After preheader insertion the predecessor map reflects the new routing:
    /// only the preheader precedes the header, and the original non-back-edge
    /// predecessor now precedes the preheader.
    #[test]
    fn test_preheader_predecessor_routing() {
        let mut f = build_simple_while();
        f.next_block_id = 10;
        let mut egraph = crate::egraph::EGraph::new();

        let loops = detect_loops(&f);
        let map = id_to_idx(&f);
        let preheader_idx = insert_preheader(&mut f, &mut egraph, &loops[0], &map);

        let (preds, _) = build_predecessor_map(&f);
        let header_idx = loops[0].header_idx; // 1

        // Header's predecessors: back-edge source (bb2) and preheader.
        let mut header_preds = preds[header_idx].clone();
        header_preds.sort();
        // bb2 is index 2, preheader_idx is 4.
        let mut expected = vec![2usize, preheader_idx];
        expected.sort();
        assert_eq!(
            header_preds, expected,
            "header must be preceded by back-edge src and preheader"
        );

        // Preheader's sole predecessor is bb0 (index 0).
        assert_eq!(
            preds[preheader_idx],
            vec![0],
            "preheader must be preceded only by bb0"
        );
    }

    // ── invariant detection ───────────────────────────────────────────────────

    /// Helper: build a simple while loop function where bb1 is the header and
    /// bb2 is the body. Returns (func, loop_info).
    fn build_while_with_egraph() -> (Function, crate::egraph::EGraph, LoopInfo) {
        use crate::egraph::ENode;
        use crate::ir::types::Type;
        use smallvec::smallvec;

        let mut f = make_func();
        let mut egraph = crate::egraph::EGraph::new();

        // bb0: entry -> bb1
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump(1));
        f.blocks.push(bb0);

        // bb1: loop header; branches on a condition
        let mut bb1 = BasicBlock::new(1, vec![]);
        // Add a BlockParam for bb1 so it shows up as loop-defined.
        let header_param = egraph.add(ENode {
            op: Op::BlockParam(1, 0, Type::I64),
            children: smallvec![],
        });
        bb1.ops.push(branch(header_param, 2, 3));
        f.blocks.push(bb1);

        // bb2: loop body -> back edge to bb1
        let mut bb2 = BasicBlock::new(2, vec![]);
        bb2.ops.push(EffectfulOp::Jump {
            target: 1,
            args: vec![header_param],
        });
        f.blocks.push(bb2);

        // bb3: exit
        let mut bb3 = BasicBlock::new(3, vec![]);
        bb3.ops.push(ret());
        f.blocks.push(bb3);

        let loops = detect_loops(&f);
        assert_eq!(loops.len(), 1);
        let loop_info = loops.into_iter().next().unwrap();

        (f, egraph, loop_info)
    }

    /// An Iconst is not in loop_defined and has no children, so it is invariant.
    #[test]
    fn test_iconst_is_invariant() {
        use crate::egraph::ENode;
        use crate::ir::types::Type;
        use smallvec::smallvec;

        let (f, mut egraph, loop_info) = build_while_with_egraph();
        let iconst = egraph.add(ENode {
            op: Op::Iconst(42, Type::I64),
            children: smallvec![],
        });
        let loop_defined = collect_loop_defined_classes(&f, &egraph, &loop_info.body);
        let mut cache = HashMap::new();
        assert!(is_class_loop_invariant(
            iconst,
            &egraph,
            &loop_defined,
            &mut cache
        ));
    }

    /// Add(function_param, iconst) — both children are outside the loop, so it
    /// is invariant.
    #[test]
    fn test_add_of_param_and_iconst_is_invariant() {
        use crate::egraph::ENode;
        use crate::ir::types::Type;
        use smallvec::smallvec;

        let (f, mut egraph, loop_info) = build_while_with_egraph();
        let param = egraph.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let iconst = egraph.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let add_node = egraph.add(ENode {
            op: Op::Add,
            children: smallvec![param, iconst],
        });
        let loop_defined = collect_loop_defined_classes(&f, &egraph, &loop_info.body);
        let mut cache = HashMap::new();
        assert!(is_class_loop_invariant(
            add_node,
            &egraph,
            &loop_defined,
            &mut cache
        ));
    }

    /// BlockParam of the loop header (bb1) must be in loop_defined and therefore
    /// NOT invariant.
    #[test]
    fn test_loop_header_block_param_is_not_invariant() {
        use crate::egraph::ENode;
        use crate::ir::types::Type;
        use smallvec::smallvec;

        let (f, mut egraph, loop_info) = build_while_with_egraph();
        // Add a BlockParam for the loop header (bb1, index 1).
        let header_id = f.blocks[loop_info.header_idx].id;
        let bp = egraph.add(ENode {
            op: Op::BlockParam(header_id, 1, Type::I64),
            children: smallvec![],
        });
        let loop_defined = collect_loop_defined_classes(&f, &egraph, &loop_info.body);
        let mut cache = HashMap::new();
        assert!(!is_class_loop_invariant(
            bp,
            &egraph,
            &loop_defined,
            &mut cache
        ));
    }

    /// LoadResult is an effectful placeholder and must NOT be invariant even if
    /// it is not in loop_defined.
    #[test]
    fn test_load_result_is_not_invariant() {
        use crate::egraph::ENode;
        use crate::ir::types::Type;
        use smallvec::smallvec;

        let (f, mut egraph, loop_info) = build_while_with_egraph();
        // Add a LoadResult node that is NOT referenced by any loop body op, so
        // collect_loop_defined_classes won't include it. Still should be rejected.
        let lr = egraph.add(ENode {
            op: Op::LoadResult(99, Type::I64),
            children: smallvec![],
        });
        let loop_defined = collect_loop_defined_classes(&f, &egraph, &loop_info.body);
        let mut cache = HashMap::new();
        assert!(!is_class_loop_invariant(
            lr,
            &egraph,
            &loop_defined,
            &mut cache
        ));
    }

    /// Add(loop_body_block_param, iconst) — one child is defined inside the loop,
    /// so the whole expression is NOT invariant.
    #[test]
    fn test_add_with_loop_defined_child_is_not_invariant() {
        use crate::egraph::ENode;
        use crate::ir::types::Type;
        use smallvec::smallvec;

        let (f, mut egraph, loop_info) = build_while_with_egraph();
        // BlockParam for the loop body block (bb2, index 2).
        let body_id = f.blocks[2].id;
        let body_bp = egraph.add(ENode {
            op: Op::BlockParam(body_id, 0, Type::I64),
            children: smallvec![],
        });
        let iconst = egraph.add(ENode {
            op: Op::Iconst(5, Type::I64),
            children: smallvec![],
        });
        let add_node = egraph.add(ENode {
            op: Op::Add,
            children: smallvec![body_bp, iconst],
        });
        let loop_defined = collect_loop_defined_classes(&f, &egraph, &loop_info.body);
        let mut cache = HashMap::new();
        assert!(!is_class_loop_invariant(
            add_node,
            &egraph,
            &loop_defined,
            &mut cache
        ));
    }
}
