use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::egraph::egraph::EGraph;
use crate::egraph::extract::ExtractionResult;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};

/// Remove blocks not reachable from the entry block via CFG edges.
///
/// All cross-block references (Jump targets, Branch targets) use BlockId values,
/// not indices into func.blocks, so `retain()` is safe; subsequent passes rebuild
/// id_to_idx maps as needed.
///
/// Returns the number of blocks removed.
pub(super) fn eliminate_unreachable_blocks(func: &mut Function) -> usize {
    if func.blocks.is_empty() {
        return 0;
    }

    // Build BlockId -> index map.
    let id_to_idx: BTreeMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // BFS from entry block (index 0).
    let mut reachable: BTreeSet<BlockId> = BTreeSet::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();

    let entry_id = func.blocks[0].id;
    reachable.insert(entry_id);
    worklist.push_back(0);

    while let Some(idx) = worklist.pop_front() {
        let block = &func.blocks[idx];
        let successors = block_successors(block.ops.last());
        for target_id in successors {
            if reachable.insert(target_id) {
                let target_idx = *id_to_idx
                    .get(&target_id)
                    .expect("branch target BlockId not in func.blocks");
                worklist.push_back(target_idx);
            }
        }
    }

    let before = func.blocks.len();

    if crate::trace::is_enabled("dce") && crate::trace::fn_matches(&func.name) {
        let removed: Vec<BlockId> = func
            .blocks
            .iter()
            .filter(|b| !reachable.contains(&b.id))
            .map(|b| b.id)
            .collect();
        if !removed.is_empty() {
            eprintln!(
                "[dce] {}: removing {} unreachable block(s): {:?}",
                func.name,
                removed.len(),
                removed
            );
        }
    }

    func.blocks.retain(|b| reachable.contains(&b.id));
    before - func.blocks.len()
}

/// Extract successor BlockIds from a terminator op.
fn block_successors(terminator: Option<&EffectfulOp>) -> Vec<BlockId> {
    match terminator {
        Some(EffectfulOp::Jump { target, .. }) => vec![*target],
        Some(EffectfulOp::Branch {
            bb_true, bb_false, ..
        }) => vec![*bb_true, *bb_false],
        _ => vec![],
    }
}

/// Fold branches with known-constant conditions into unconditional jumps.
///
/// Branch conditions are Flags-typed from Icmp/Fcmp. After isel, the e-class
/// may also contain X86Sub/Proj1 nodes. We scan the cond e-class for an Icmp
/// node whose children are both constants, then evaluate the comparison using
/// the Branch's condition code.
///
/// Returns the number of branches folded.
pub(super) fn fold_constant_branches(
    func: &mut Function,
    egraph: &EGraph,
    _extraction: &ExtractionResult,
) -> usize {
    let mut folded = 0;

    for block in func.blocks.iter_mut() {
        let Some(terminator) = block.ops.last_mut() else {
            continue;
        };

        if let EffectfulOp::Branch {
            cond,
            cc,
            bb_true,
            bb_false,
            ref true_args,
            ref false_args,
        } = *terminator
        {
            let canon_cond = egraph.unionfind.find_immutable(cond);

            // Scan the cond e-class for an Icmp node with constant children.
            // The extraction may have picked X86Sub/Proj1 over Icmp, but the
            // Icmp node is still in the class and carries the comparison operands.
            let takes_true = try_eval_branch_cond(egraph, canon_cond, cc);

            let Some(takes_true) = takes_true else {
                continue;
            };

            if crate::trace::is_enabled("dce") {
                eprintln!(
                    "[dce] fold branch in bb{}: cc={:?} -> {}",
                    block.id,
                    cc,
                    if takes_true { "true" } else { "false" }
                );
            }

            *terminator = if takes_true {
                EffectfulOp::Jump {
                    target: bb_true,
                    args: true_args.clone(),
                }
            } else {
                EffectfulOp::Jump {
                    target: bb_false,
                    args: false_args.clone(),
                }
            };
            folded += 1;
        }
    }

    folded
}

/// Try to statically evaluate a branch condition by scanning the e-class for
/// an Icmp node with constant children.
fn try_eval_branch_cond(
    egraph: &EGraph,
    cond_class: ClassId,
    branch_cc: crate::ir::condcode::CondCode,
) -> Option<bool> {
    let class = &egraph.classes[cond_class.0 as usize];
    for node in &class.nodes {
        if let Op::Icmp(_) = &node.op
            && node.children.len() == 2
        {
            let lhs = egraph.unionfind.find_immutable(node.children[0]);
            let rhs = egraph.unionfind.find_immutable(node.children[1]);
            if let (Some((a, _)), Some((b, _))) =
                (egraph.get_constant(lhs), egraph.get_constant(rhs))
            {
                return crate::egraph::algebraic::eval_icmp(&branch_cc, a, b);
            }
        }
    }
    None
}

/// Collect all ClassIds that are consumed (used as inputs) by effectful ops.
///
/// Seeds from effectful op input fields (addr, val, cond, args, ret val),
/// then transitively walks extracted node children. Load `result` ClassIds
/// are NOT seeded because they are outputs, not inputs.
fn collect_consumed_class_ids(
    func: &Function,
    egraph: &EGraph,
    extraction: &ExtractionResult,
) -> BTreeSet<ClassId> {
    let mut seeds: Vec<ClassId> = Vec::new();

    for block in &func.blocks {
        for op in &block.ops {
            match op {
                EffectfulOp::Load { addr, .. } => {
                    // addr is an input; result is an output (not seeded)
                    seeds.push(*addr);
                }
                EffectfulOp::Store { addr, val, .. } => {
                    seeds.push(*addr);
                    seeds.push(*val);
                }
                EffectfulOp::Call { args, .. } => {
                    // args are inputs; results are outputs (not seeded)
                    seeds.extend(args.iter().copied());
                }
                EffectfulOp::Branch {
                    cond,
                    true_args,
                    false_args,
                    ..
                } => {
                    seeds.push(*cond);
                    seeds.extend(true_args.iter().copied());
                    seeds.extend(false_args.iter().copied());
                }
                EffectfulOp::Jump { args, .. } => {
                    seeds.extend(args.iter().copied());
                }
                EffectfulOp::Ret { val } => {
                    if let Some(v) = val {
                        seeds.push(*v);
                    }
                }
            }
        }
    }

    // Canonicalize all seeds.
    for seed in &mut seeds {
        *seed = egraph.unionfind.find_immutable(*seed);
    }

    // Transitively walk extraction children.
    let mut consumed: BTreeSet<ClassId> = BTreeSet::new();
    let mut worklist: VecDeque<ClassId> = seeds.into_iter().collect();

    while let Some(cid) = worklist.pop_front() {
        if !consumed.insert(cid) {
            continue; // already visited
        }
        if let Some(extracted) = extraction.choices.get(&cid) {
            for &child in &extracted.children {
                if child != ClassId::NONE {
                    let canon = egraph.unionfind.find_immutable(child);
                    if !consumed.contains(&canon) {
                        worklist.push_back(canon);
                    }
                }
            }
        }
    }

    consumed
}

/// Remove Load effectful ops whose result ClassId is not consumed by any
/// other effectful op in the function.
///
/// Stores and Calls are never eliminated (they have side effects).
/// A Load is considered dead when its canonical result ClassId does not
/// appear in the consumed set built by `collect_consumed_class_ids`.
///
/// Returns the number of loads eliminated.
pub(super) fn eliminate_dead_loads(
    func: &mut Function,
    egraph: &EGraph,
    extraction: &ExtractionResult,
) -> usize {
    let consumed = collect_consumed_class_ids(func, egraph, extraction);
    let mut eliminated = 0;

    for block in func.blocks.iter_mut() {
        // Only scan non-terminator ops (terminators are never loads).
        // Use `i + 1 < block.ops.len()` so the bound stays correct after removals.
        let mut i = 0;
        while i + 1 < block.ops.len() {
            if let EffectfulOp::Load { result, .. } = &block.ops[i] {
                let canon_result = egraph.unionfind.find_immutable(*result);
                if !consumed.contains(&canon_result) {
                    if crate::trace::is_enabled("dce") {
                        eprintln!(
                            "[dce] eliminate dead load in bb{}: result class {}",
                            block.id, canon_result.0
                        );
                    }
                    block.ops.remove(i);
                    eliminated += 1;
                    // term_idx shifts down by 1; don't increment i
                    continue;
                }
            }
            i += 1;
        }
    }

    eliminated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::function::{BasicBlock, Function};

    fn make_function(name: &str, blocks: Vec<BasicBlock>) -> Function {
        let max_id = blocks.iter().map(|b| b.id).max().unwrap_or(0);
        Function {
            name: name.to_string(),
            param_types: vec![],
            return_types: vec![],
            blocks,
            param_class_ids: vec![],
            egraph: None,
            stack_slots: vec![],
            noinline: false,
            next_block_id: max_id + 1,
        }
    }

    fn make_block(id: BlockId, ops: Vec<EffectfulOp>) -> BasicBlock {
        BasicBlock {
            id,
            param_types: vec![],
            ops,
        }
    }

    #[test]
    fn test_eliminate_unreachable_linear() {
        let mut func = make_function(
            "test",
            vec![
                make_block(
                    0,
                    vec![EffectfulOp::Jump {
                        target: 1,
                        args: vec![],
                    }],
                ),
                make_block(
                    1,
                    vec![EffectfulOp::Jump {
                        target: 2,
                        args: vec![],
                    }],
                ),
                make_block(2, vec![EffectfulOp::Ret { val: None }]),
            ],
        );
        let removed = eliminate_unreachable_blocks(&mut func);
        assert_eq!(removed, 0);
        assert_eq!(func.blocks.len(), 3);
    }

    #[test]
    fn test_eliminate_unreachable_one_dead() {
        let mut func = make_function(
            "test",
            vec![
                make_block(
                    0,
                    vec![EffectfulOp::Jump {
                        target: 2,
                        args: vec![],
                    }],
                ),
                make_block(1, vec![EffectfulOp::Ret { val: None }]), // unreachable
                make_block(2, vec![EffectfulOp::Ret { val: None }]),
            ],
        );
        let removed = eliminate_unreachable_blocks(&mut func);
        assert_eq!(removed, 1);
        assert_eq!(func.blocks.len(), 2);
        assert_eq!(func.blocks[0].id, 0);
        assert_eq!(func.blocks[1].id, 2);
    }

    #[test]
    fn test_eliminate_unreachable_loop() {
        let mut func = make_function(
            "test",
            vec![
                make_block(
                    0,
                    vec![EffectfulOp::Jump {
                        target: 1,
                        args: vec![],
                    }],
                ),
                make_block(
                    1,
                    vec![EffectfulOp::Jump {
                        target: 1,
                        args: vec![],
                    }],
                ), // self-loop
            ],
        );
        let removed = eliminate_unreachable_blocks(&mut func);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_eliminate_unreachable_diamond() {
        use crate::ir::condcode::CondCode;
        let mut func = make_function(
            "test",
            vec![
                make_block(
                    0,
                    vec![EffectfulOp::Branch {
                        cond: ClassId(0),
                        cc: CondCode::Ne,
                        bb_true: 1,
                        bb_false: 2,
                        true_args: vec![],
                        false_args: vec![],
                    }],
                ),
                make_block(
                    1,
                    vec![EffectfulOp::Jump {
                        target: 3,
                        args: vec![],
                    }],
                ),
                make_block(
                    2,
                    vec![EffectfulOp::Jump {
                        target: 3,
                        args: vec![],
                    }],
                ),
                make_block(3, vec![EffectfulOp::Ret { val: None }]),
            ],
        );
        let removed = eliminate_unreachable_blocks(&mut func);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_eliminate_unreachable_single_block() {
        let mut func = make_function(
            "test",
            vec![make_block(0, vec![EffectfulOp::Ret { val: None }])],
        );
        let removed = eliminate_unreachable_blocks(&mut func);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_eliminate_unreachable_chain() {
        let mut func = make_function(
            "test",
            vec![
                make_block(0, vec![EffectfulOp::Ret { val: None }]),
                make_block(
                    1,
                    vec![EffectfulOp::Jump {
                        target: 2,
                        args: vec![],
                    }],
                ), // unreachable
                make_block(
                    2,
                    vec![EffectfulOp::Jump {
                        target: 3,
                        args: vec![],
                    }],
                ), // unreachable
                make_block(3, vec![EffectfulOp::Ret { val: None }]), // unreachable
            ],
        );
        let removed = eliminate_unreachable_blocks(&mut func);
        assert_eq!(removed, 3);
        assert_eq!(func.blocks.len(), 1);
    }
}

/// DCE pass 1: unreachable block elimination after inlining, before LICM.
pub(super) fn run_dce1(func: &mut Function) {
    eliminate_unreachable_blocks(func);
}

/// DCE pass 2: constant branch folding + unreachable blocks + dead loads.
///
/// Runs after e-graph extraction, before the immutable func freeze and
/// index construction in compile().
fn run_dce2_inner(func: &mut Function, egraph: &EGraph, extraction: &ExtractionResult) {
    let folded = fold_constant_branches(func, egraph, extraction);
    let unreachable = eliminate_unreachable_blocks(func);
    let dead_loads = eliminate_dead_loads(func, egraph, extraction);

    if (folded > 0 || unreachable > 0 || dead_loads > 0)
        && crate::trace::is_enabled("dce")
        && crate::trace::fn_matches(&func.name)
    {
        eprintln!(
            "[dce] dce2 {}: folded {} branch(es), removed {} unreachable block(s), eliminated {} dead load(s)",
            func.name, folded, unreachable, dead_loads
        );
    }
}

/// Run DCE2 and remap LICM extra_roots through the block removal.
///
/// Extra roots are keyed by block index; block removal shifts indices.
/// This converts to BlockId keys, runs DCE2, then rebuilds index keys.
pub(super) fn run_dce2_with_extra_roots(
    func: &mut Function,
    egraph: &EGraph,
    extraction: &ExtractionResult,
    extra_roots: super::licm::ExtraRoots,
) -> super::licm::ExtraRoots {
    // Filter extra_roots: LICM's find_invariant_classes walks the PRE-saturation
    // e-graph transitively and hoists any invariant ancestor. After saturation
    // and extraction, the chosen ops may no longer reference those ancestors
    // (e.g. `i * 4` may fold into an Addr with scale=4 embedded, leaving
    // iconst(4, I64) orphan in the e-graph). Emitting an orphan VRegInst for
    // such a class clobbers a live register at the preheader with no consumer.
    //
    // Keep only hoisted classes that are reachable from effectful-op operands
    // via extraction.choices (i.e., classes the extracted IR actually uses).
    let consumed = collect_consumed_class_ids(func, egraph, extraction);
    let filtered_extra_roots: super::licm::ExtraRoots = extra_roots
        .into_iter()
        .map(|(idx, classes)| {
            let kept: Vec<ClassId> = classes
                .into_iter()
                .filter(|cid| {
                    let canon = egraph.unionfind.find_immutable(*cid);
                    consumed.contains(&canon)
                })
                .collect();
            (idx, kept)
        })
        .filter(|(_, classes)| !classes.is_empty())
        .collect();

    // Convert index-keyed extra_roots to BlockId-keyed.
    let mut id_keyed: BTreeMap<BlockId, Vec<ClassId>> = BTreeMap::new();
    for (&idx, roots) in &filtered_extra_roots {
        debug_assert!(
            idx < func.blocks.len(),
            "extra_roots index {idx} out of bounds (blocks.len() = {})",
            func.blocks.len()
        );
        if idx < func.blocks.len() {
            id_keyed.insert(func.blocks[idx].id, roots.clone());
        }
    }

    run_dce2_inner(func, egraph, extraction);

    // Rebuild index-keyed map using post-DCE2 block order.
    let mut rebuilt: BTreeMap<usize, Vec<ClassId>> = BTreeMap::new();
    for (i, block) in func.blocks.iter().enumerate() {
        if let Some(roots) = id_keyed.remove(&block.id) {
            rebuilt.insert(i, roots);
        }
    }
    rebuilt
}
