//! Block parameters every predecessor proves constant, removed before saturation.
//!
//! A parameter whose predecessors all pass the same constant *is* that constant:
//! the meet of the incoming values, where two constants that differ and anything
//! the analysis cannot prove constant both give up. SSA construction cannot see
//! that, because it creates a parameter for every variable live across an edge
//! and decides nothing about the values.
//!
//! # What it is not
//!
//! The *conditional* half of SCCP is missing: an edge a constant branch can
//! never take still contributes its argument to the meet here, so a parameter
//! two arms disagree about survives even when one arm is unreachable. The blocks
//! themselves do go -- `dce`'s constant branch folding removes them -- but that
//! runs after extraction, too late to feed this.
//!
//! # Why this removes the parameter rather than merging it
//!
//! Merging the parameter's class with the constant's is the obvious form and it
//! miscompiles. The merged class holds *both* `BlockParam(b, i)` and
//! `Iconst(k)`, and [`super::cfg::resolve_block_param_vreg`] asks the target
//! block's own `BlockParam` first -- so a use inside the block reads the
//! parameter's register, whose phi copy on the edge sources that same merged
//! class and therefore copies the register from itself. The register holds
//! whatever it held before the copy.
//!
//! Removing the parameter from the block, from every incoming edge, and from the
//! e-graph leaves the class holding the constant alone, so there is no second
//! answer for anything to prefer. That is also why the union is safe here: the
//! block's own uses still name the parameter's class, and after the removal that
//! class has the constant's node and nothing else.
//!
//! # Why it runs before saturation
//!
//! The copies are the smaller half. What the removal buys is that the rules then
//! see a constant where they saw an opaque parameter, so `x * 3 + 1` behind a
//! parameter every arm passes `7` folds to `22` instead of being computed. After
//! saturation there is nothing left to fold.
//!
//! # Why it needs no acceptance test
//!
//! [`super::phi_removal`] removes on the CFG and then checks its own result,
//! because the class it merges into may be one no block can re-emit -- a
//! `LoadResult` re-named where its emitter does not dominate mints a second
//! barrier instruction for one effectful op. A constant is re-emittable
//! anywhere, at the cost of one instruction, which is the whole of what that
//! test is protecting. This pass also runs before extraction, so there are no
//! schedules for it to invalidate.

use std::collections::BTreeMap;

use smallvec::smallvec;

use crate::egraph::EGraph;
use crate::egraph::enode::ENode;
use crate::ir::Type;
use crate::ir::effectful::BlockId;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op, PureOp};

use super::barrier::{terminator_edges, terminator_edges_mut};

/// A parameter position the meet proved constant, and the value it carries.
struct ConstParam {
    block: BlockId,
    pidx: u32,
    value: i64,
    ty: Type,
}

/// Remove every block parameter the meet proves constant, to a fixpoint.
///
/// Returns whether anything was removed, which is what tells the caller to
/// canonicalize the CFG's `ClassId`s and re-verify.
///
/// Iterating matters: a removed parameter becomes a constant its own block may
/// pass on, so a chain of blocks handing one value along collapses one link per
/// round. The number of parameters is a bound on the rounds -- each round
/// removes at least one -- so the loop needs no other cap.
pub(super) fn run_sccp(func: &mut Function, egraph: &mut EGraph) -> bool {
    let mut changed = false;
    loop {
        let found = find_constant_params(func, egraph);
        if found.is_empty() {
            return changed;
        }
        apply(func, egraph, &found);
        changed = true;
    }
}

/// Every parameter position whose incoming values meet to one constant.
fn find_constant_params(func: &Function, egraph: &EGraph) -> Vec<ConstParam> {
    // Per (target, position): the constants its predecessors pass, and how many
    // edges supplied a value at all. `None` once any predecessor passes
    // something the analysis cannot prove constant.
    let mut incoming: BTreeMap<(BlockId, u32), Option<(i64, Type)>> = BTreeMap::new();
    let mut self_edges: BTreeMap<(BlockId, u32), usize> = BTreeMap::new();
    let mut supplied: BTreeMap<(BlockId, u32), usize> = BTreeMap::new();
    let mut pred_count: BTreeMap<BlockId, usize> = BTreeMap::new();

    let block_param_map = egraph.block_param_classes();
    let param_class = |block: BlockId, pidx: u32| {
        block_param_map
            .get(&(block, pidx))
            .map(|&c| egraph.find_immutable(c))
    };

    for block in &func.blocks {
        let Some(term) = block.ops.last() else {
            continue;
        };
        for (target, args) in terminator_edges(term) {
            *pred_count.entry(target).or_insert(0) += 1;
            for (pidx, &arg) in args.expect_classes().iter().enumerate() {
                let key = (target, pidx as u32);
                let arg = egraph.find_immutable(arg);
                // An argument that *is* the parameter carries no second opinion:
                // `p = phi(k, p)` is `k` on entry and `p` thereafter, so the
                // self-reference constrains nothing. Without this, no
                // loop-carried constant is ever provable.
                if param_class(target, pidx as u32) == Some(arg) {
                    *self_edges.entry(key).or_insert(0) += 1;
                    continue;
                }
                *supplied.entry(key).or_insert(0) += 1;
                let met = incoming
                    .entry(key)
                    .or_insert_with(|| egraph.get_constant(arg));
                if *met != egraph.get_constant(arg) {
                    *met = None;
                }
            }
        }
    }

    let entry = func.blocks.first().map(|b| b.id);
    let mut found = Vec::new();
    for block in &func.blocks {
        // The entry block's parameters are the function's own: no edge supplies
        // them and nothing may replace them.
        if Some(block.id) == entry {
            continue;
        }
        let preds = pred_count.get(&block.id).copied().unwrap_or(0);
        if preds == 0 {
            continue;
        }
        for pidx in 0..block.param_types.len() as u32 {
            let key = (block.id, pidx);
            let supplied = supplied.get(&key).copied().unwrap_or(0);
            let selfs = self_edges.get(&key).copied().unwrap_or(0);
            // An arity the predecessors disagree about is a broken CFG, not
            // something to repair here; `verify` is where that is reported. A
            // position every edge self-references has no incoming value.
            if supplied + selfs != preds || supplied == 0 {
                continue;
            }
            let Some(Some((value, ty))) = incoming.get(&key).cloned() else {
                continue;
            };
            // The node this mints is unioned with the parameter's class, and
            // `EGraph::merge` requires the two classes to agree on type.
            if ty != block.param_types[pidx as usize] {
                continue;
            }
            found.push(ConstParam {
                block: block.id,
                pidx,
                value,
                ty,
            });
        }
    }

    // Two parameter positions can name one class, and then they are one value:
    // the union below would put both constants in that class, leaving extraction
    // to pick one and the analysis to report neither. Positions that share a
    // class and disagree describe a CFG whose own edges contradict each other,
    // so none of them is safe to act on.
    let mut by_class: BTreeMap<ClassId, Vec<usize>> = BTreeMap::new();
    for (i, p) in found.iter().enumerate() {
        if let Some(cid) = param_class(p.block, p.pidx) {
            by_class.entry(cid).or_default().push(i);
        }
    }
    let conflicted: Vec<usize> = by_class
        .values()
        .filter(|group| {
            group.iter().any(|&i| {
                (found[i].value, &found[i].ty) != (found[group[0]].value, &found[group[0]].ty)
            })
        })
        .flatten()
        .copied()
        .collect();
    let mut idx = 0usize;
    found.retain(|_| {
        let keep = !conflicted.contains(&idx);
        idx += 1;
        keep
    });
    found
}

/// Union each proved parameter with its constant, then drop the position from
/// the block, from the `BlockParam` nodes, and from every incoming edge.
///
/// The constant is added rather than taken from a predecessor's argument class:
/// the analysis can prove a class constant without the class holding an
/// `Iconst` node, and extraction can only emit a node that exists.
fn apply(func: &mut Function, egraph: &mut EGraph, found: &[ConstParam]) {
    let block_param_map = egraph.block_param_classes();
    egraph.under_rule("sccp", |egraph| {
        for p in found {
            let konst = egraph.add(ENode {
                op: Op::Pure(PureOp::Iconst(p.value, p.ty.clone())),
                children: smallvec![],
            });
            if let Some(&param) = block_param_map.get(&(p.block, p.pidx)) {
                egraph.merge(param, konst);
            }
        }
        true
    });
    egraph.rebuild();

    // Which positions each touched block keeps, in its old numbering.
    let mut keep: BTreeMap<BlockId, Vec<u32>> = BTreeMap::new();
    for p in found {
        keep.entry(p.block).or_insert_with(|| {
            let block = func
                .blocks
                .iter()
                .find(|b| b.id == p.block)
                .expect("parameter's block");
            (0..block.param_types.len() as u32).collect()
        });
    }
    for p in found {
        keep.get_mut(&p.block)
            .expect("seeded above")
            .retain(|&pidx| pidx != p.pidx);
    }
    egraph.rewrite_block_params(&keep);

    for block in func.blocks.iter_mut() {
        if let Some(kept) = keep.get(&block.id) {
            let mut pidx = 0u32;
            block.param_types.retain(|_| {
                let keep_this = kept.contains(&pidx);
                pidx += 1;
                keep_this
            });
        }
        let Some(term) = block.ops.last_mut() else {
            continue;
        };
        for (target, args) in terminator_edges_mut(term) {
            let Some(kept) = keep.get(&target) else {
                continue;
            };
            let classes = args.expect_classes_mut();
            let mut pidx = 0u32;
            classes.retain(|_| {
                let keep_this = kept.contains(&pidx);
                pidx += 1;
                keep_this
            });
        }
    }
}
