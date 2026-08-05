//! Redundant block parameter elimination.
//!
//! A block parameter is a phi. A phi whose operands are all one value, ignoring
//! its own self-references, is that value -- Braun et al.'s `tryRemoveTrivialPhi`,
//! the step this backend's SSA construction never had. `read_variable` creates a
//! parameter before recursing into predecessors, to break loop cycles, and then
//! keeps it whatever the operands turn out to be.
//!
//! The self-reference rule is what makes this reach loops at all: a value carried
//! around a loop and never reassigned gives `p = phi(p_init, p)`, whose operands
//! reduce to `{p_init}`. Without the rule nothing in a loop is ever removable.
//!
//! Measured on the generated corpus: **85-94% of block parameters are redundant.**
//! One loop header carried 28 parameters where 4 were real. The cost of the other
//! 24 is paid three times over -- a register-to-register copy per parameter per
//! incoming edge in the parallel copy, a 24-wide clique in the interference graph
//! (`add_block_param_interferences` asserts one, correctly, because the copies are
//! simultaneous), and, once the splitter routes what cannot be coloured, a store
//! and a reload per parameter per iteration for a value nothing reads.
//!
//! Removing them is a canonicalization, not a heuristic: it removes work rather
//! than moving it.
//!
//! In an e-graph the "replace all uses" half is free -- union the parameter's class
//! with its single source and every reference resolves through the union-find. What
//! costs something is the parameter *positions*: removing one renumbers the rest,
//! and the `Op::BlockParam(block, index, ty)` nodes carry that index. See
//! [`EGraph::rewrite_block_params`] for why that renumbering has to happen in one
//! pass over a drained memo.
//!
//! # Incomplete: `enable_phi_simplify` is off by default
//!
//! **One e-class is not one value, and that is the whole difficulty.** An e-class is
//! a pure *expression*; two predecessors computing the same expression share a class
//! while each emits it into a register of its own, which is exactly what the phi
//! reconciles. Removing it there leaves the block reading a register only one path
//! wrote, and the machine verifier says so outright -- "reads XMM8 on a path where
//! nothing writes it", three times on `pressure` seed 22.
//!
//! Requiring every supplying predecessor to *dominate* the block is sound and too
//! strong: it rejects the removals that matter (seed 22 goes back to
//! `gpr_overshoot=4`) and, measured, two programs still come out wrong at -O1 with
//! both verifiers silent. So the condition is not just dominance either.
//!
//! What the condition has to express is "the value is in one register on every path
//! that reaches this block", which is a statement about the *extraction*, not about
//! the CFG -- and extraction runs later. The likely shape of the fix is to let the
//! class be re-emitted per block as it already can be (`class_emitted_in` and the
//! per-block snapshots exist for this), and to remove the parameter only where the
//! value is not re-emitted; or to run this after extraction, where the emission is
//! decided, and pay for rebuilding everything keyed on parameter index.
//!
//! Kept because the measurement is worth having: 85-94% redundant across the whole
//! generated corpus, and one loop header carrying 28 parameters where 4 are real.
//!
//! **This module is superseded rather than fixed.** `docs/refactor-roadmap.md` step 2
//! replaces it with the same rule over VRegs, where a phi operand is a *def* and the
//! difficulty above does not exist. Do not re-attempt it on `ClassId`s.

use std::collections::{BTreeMap, BTreeSet};

use super::barrier::terminator_edges;
use super::cfg::{block_id_to_idx, compute_idom, compute_rpo, dominates};
use crate::egraph::EGraph;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::ClassId;

/// Eliminate every block parameter that is a trivial phi. Returns how many went.
///
/// Runs before `block_param_map` and linearization: everything downstream is keyed
/// on `(BlockId, param_index)`, so the arity has to be final before any of it is
/// built.
pub fn simplify_block_params(func: &mut Function, egraph: &mut EGraph) -> usize {
    let Some(entry) = func.blocks.first().map(|b| b.id) else {
        return 0;
    };

    // Which class each parameter position names. A position with no `BlockParam`
    // node cannot be proven trivial -- nothing names the parameter, so there is no
    // class to union its source into.
    let param_class = egraph.block_param_classes();

    // The arguments every predecessor passes to each position, tagged with which
    // predecessor passed them.
    let mut incoming: BTreeMap<(BlockId, u32), Vec<(BlockId, ClassId)>> = BTreeMap::new();
    let mut pred_count: BTreeMap<BlockId, usize> = BTreeMap::new();
    for block in &func.blocks {
        let Some(term) = block.ops.last() else {
            continue;
        };
        for (target, args) in terminator_edges(term) {
            *pred_count.entry(target).or_insert(0) += 1;
            for (i, &arg) in args.expect_classes().iter().enumerate() {
                incoming
                    .entry((target, i as u32))
                    .or_default()
                    .push((block.id, arg));
            }
        }
    }

    let block_idx = block_id_to_idx(func);
    let idom = compute_idom(func, &compute_rpo(func));

    // A block whose predecessors disagree about arity, or whose parameters no edge
    // supplies, is left alone: the arity is a precondition here, not something to
    // repair. `verify` is the place that reports it.
    let arity: BTreeMap<BlockId, usize> = func
        .blocks
        .iter()
        .map(|b| (b.id, b.param_types.len()))
        .collect();
    let malformed: BTreeSet<BlockId> = func
        .blocks
        .iter()
        .filter(|b| {
            let n = b.param_types.len();
            let preds = pred_count.get(&b.id).copied().unwrap_or(0);
            (0..n as u32).any(|i| {
                incoming.get(&(b.id, i)).map_or(n > 0, |v| v.len() != preds) && b.id != entry
            })
        })
        .map(|b| b.id)
        .collect();

    // Fixpoint: removing one trivial phi can leave another with a single source.
    let mut trivial: BTreeSet<(BlockId, u32)> = BTreeSet::new();
    loop {
        let mut progressed = false;
        for (&(bid, pidx), &cls) in &param_class {
            // The entry block's parameters are the function's own: no edge supplies
            // them and nothing may replace them.
            if bid == entry || malformed.contains(&bid) || trivial.contains(&(bid, pidx)) {
                continue;
            }
            if pidx as usize >= arity.get(&bid).copied().unwrap_or(0) {
                continue;
            }
            let Some(srcs) = incoming.get(&(bid, pidx)) else {
                continue;
            };
            let cls_canon = egraph.find_immutable(cls);
            let supplying: Vec<(BlockId, ClassId)> = srcs
                .iter()
                .map(|&(from, s)| (from, egraph.find_immutable(s)))
                .filter(|&(_, s)| s != cls_canon)
                .collect();
            let distinct: BTreeSet<ClassId> = supplying.iter().map(|&(_, s)| s).collect();
            if distinct.len() != 1 {
                continue;
            }
            let source = *distinct.iter().next().expect("one element");
            // One class is not one value. An e-class is a pure *expression*, and
            // two predecessors computing the same expression share a class while
            // each emits it into a register of its own -- which is what the phi
            // exists to reconcile. Dropping it there leaves the block reading a
            // register only one path wrote, and the machine verifier says so:
            // "reads XMM8 on a path where nothing writes it".
            //
            // Sound when every predecessor that supplies the value dominates this
            // block, because then the value is on the one path that reaches it. The
            // two shapes worth having both satisfy it: a single-predecessor
            // pass-through block, and a loop-carried value whose latch edge passes
            // the parameter itself and so drops out as a self-reference.
            let Some(&target_idx) = block_idx.get(&bid) else {
                continue;
            };
            let all_dominate = supplying.iter().all(|(from, _)| {
                block_idx
                    .get(from)
                    .is_some_and(|&fi| dominates(fi, target_idx, &idom))
            });
            if !all_dominate {
                continue;
            }
            // Types must match to merge; a parameter typed differently from its
            // argument is malformed IR and `verify` reports it.
            if egraph.class(cls_canon).ty != egraph.class(source).ty {
                continue;
            }
            egraph.merge(cls_canon, source);
            trivial.insert((bid, pidx));
            progressed = true;
        }
        if !progressed {
            break;
        }
        egraph.rebuild();
    }
    if trivial.is_empty() {
        return 0;
    }
    egraph.rebuild();

    // Surviving positions, in order, per block that lost any.
    let touched: BTreeSet<BlockId> = trivial.iter().map(|&(b, _)| b).collect();
    let keep: BTreeMap<BlockId, Vec<u32>> = touched
        .iter()
        .map(|&bid| {
            let n = arity.get(&bid).copied().unwrap_or(0) as u32;
            let survivors = (0..n).filter(|i| !trivial.contains(&(bid, *i))).collect();
            (bid, survivors)
        })
        .collect();

    for block in func.blocks.iter_mut() {
        if let Some(survivors) = keep.get(&block.id) {
            block.param_types = survivors
                .iter()
                .map(|&i| block.param_types[i as usize].clone())
                .collect();
        }
        if let Some(term) = block.ops.last_mut() {
            retain_edge_args(term, &keep);
        }
    }

    egraph.rewrite_block_params(&keep);
    trivial.len()
}

/// Drop the arguments feeding removed positions, per edge.
fn retain_edge_args(term: &mut EffectfulOp, keep: &BTreeMap<BlockId, Vec<u32>>) {
    let filter = |target: &BlockId, args: &mut Vec<ClassId>| {
        if let Some(survivors) = keep.get(target) {
            *args = survivors
                .iter()
                .filter_map(|&i| args.get(i as usize).copied())
                .collect();
        }
    };
    match term {
        EffectfulOp::Jump { target, args } => filter(target, args.expect_classes_mut()),
        EffectfulOp::Branch {
            bb_true,
            bb_false,
            true_args,
            false_args,
            ..
        } => {
            filter(bb_true, true_args.expect_classes_mut());
            filter(bb_false, false_args.expect_classes_mut());
        }
        _ => {}
    }
}
