//! Trivial block parameters, removed on the CFG and paid for by re-linearizing.
//!
//! A block parameter reconciles predecessors that computed the same value into
//! different registers. When they did not -- when every predecessor passes the
//! *same* VReg -- the parameter reconciles nothing: it costs a copy on each
//! incoming edge, a slot in the parameter clique the allocator must colour, and
//! a value the splitter counts as live across the block's whole marker run.
//!
//! # Why this removes on the CFG and re-runs linearization
//!
//! Removing a parameter means the block names the class directly, and the class
//! must therefore be *available* in that block: either its emitter dominates the
//! block, or the block emits it again. That decision belongs to linearization
//! and is made nowhere else, which is why the removal hands the reduced CFG back
//! rather than patching linearization's output. `src/compile/phi_simplify.rs`
//! tried the patch twice and its module doc records both failures -- the block
//! read a register only one path had written.
//!
//! # Tier 1 only, so far
//!
//! Every predecessor passing the same VReg needs no justification beyond that:
//! the value is already in one register on every path into the block, so nothing
//! has to be re-emitted and no cost model has to agree. Predecessors that pass
//! *different* VRegs of the same class are the case the parameter exists for;
//! removing those trades copies for a recomputation and wants the cost model,
//! which is step 2's tier 2 in `docs/refactor-roadmap.md`.

use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::EGraph;
use crate::egraph::extract::{ExtractionResult, VReg};
use crate::ir::effectful::BlockId;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};

use super::barrier::{terminator_edges, terminator_edges_mut};
use super::cfg::dominates;
use super::linearize::Linearized;

/// Which parameter positions each block keeps, for the blocks that lose any.
pub(super) struct Removal {
    /// `keep[block]` lists the surviving positions in their old numbering, which
    /// is the form [`crate::egraph::EGraph::rewrite_block_params`] renumbers by.
    pub keep: BTreeMap<BlockId, Vec<u32>>,
    /// How many parameters went, for the trace line.
    pub removed: usize,
    /// For each removed parameter, the class it named and the class its
    /// predecessors pass. The two must be unioned before the parameter's node
    /// goes: every use inside the block still names the parameter's class, and a
    /// class whose only node was that parameter has nothing left to lower.
    pub merges: Vec<(ClassId, ClassId)>,
}

impl Removal {
    pub fn is_empty(&self) -> bool {
        self.removed == 0
    }
}

/// Find every block parameter whose predecessors all pass one VReg.
///
/// Reads committed arguments, so it must run after
/// [`super::cfg::commit_terminator_arg_vregs`] and before anything consumes the
/// schedules -- the removal invalidates them.
///
/// A parameter the latch passes back to itself is a self-reference, not a second
/// opinion: `p = phi(p_init, p)` carries one value, and ignoring the operand that
/// *is* the parameter is what lets a loop-carried invariant go. Without it
/// nothing inside a loop is ever removable.
pub(super) fn find_trivial_params(
    func: &Function,
    egraph: &EGraph,
    lin: &Linearized,
    extraction: &ExtractionResult,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
) -> Removal {
    let block_param_vregs = &lin.block_param_vregs;
    let entry = func.blocks.first().map(|b| b.id);

    // Per (target, position): the VRegs its predecessors pass, the class they
    // carry, and how many edges supplied one.
    let mut incoming: BTreeMap<(BlockId, u32), (BTreeSet<VReg>, BTreeSet<ClassId>, usize)> =
        BTreeMap::new();
    let mut pred_count: BTreeMap<BlockId, usize> = BTreeMap::new();
    for block in &func.blocks {
        let Some(term) = block.ops.last() else {
            continue;
        };
        for (target, args) in terminator_edges(term) {
            *pred_count.entry(target).or_insert(0) += 1;
            let Some(committed) = args.as_committed() else {
                // Before the commit there is nothing to compare; the caller runs
                // this after it, so this is a malformed edge rather than a stage.
                continue;
            };
            for (pidx, arg) in committed.iter().enumerate() {
                let slot = incoming.entry((target, pidx as u32)).or_default();
                slot.0.insert(arg.vreg);
                slot.1.insert(egraph.unionfind.find_immutable(arg.class));
                slot.2 += 1;
            }
        }
    }

    let mut keep: BTreeMap<BlockId, Vec<u32>> = BTreeMap::new();
    let mut merges: Vec<(ClassId, ClassId)> = Vec::new();
    let mut removed = 0usize;
    for (block_idx, block) in func.blocks.iter().enumerate() {
        // The entry block's parameters are the function's own: no edge supplies
        // them and nothing may replace them.
        if Some(block.id) == entry {
            continue;
        }
        let preds = pred_count.get(&block.id).copied().unwrap_or(0);
        if preds == 0 {
            continue;
        }
        let mut survivors: Vec<u32> = Vec::with_capacity(block.param_types.len());
        let mut dropped_any = false;
        for pidx in 0..block.param_types.len() as u32 {
            let Some((vregs, classes, edges)) = incoming.get(&(block.id, pidx)) else {
                survivors.push(pidx);
                continue;
            };
            // An arity the predecessors disagree about is a broken CFG, not
            // something to repair here. `verify` is where that is reported.
            if *edges != preds {
                survivors.push(pidx);
                continue;
            }
            let own = block_param_vregs.get(&(block.id, pidx)).copied();
            let own_class = block_param_map
                .get(&(block.id, pidx))
                .map(|&c| egraph.unionfind.find_immutable(c));
            let mut distinct = vregs.clone();
            if let Some(own) = own {
                distinct.remove(&own);
            }
            let mut source_classes = classes.clone();
            if let Some(own_class) = own_class {
                source_classes.remove(&own_class);
            }
            // One register on every path, and one class naming it. The class
            // check is not redundant: it is what the merge below needs, and a
            // parameter whose class nothing names cannot be merged away.
            match (distinct.len(), source_classes.len(), own_class) {
                (1, 1, Some(own_class)) => {
                    let src = *source_classes.iter().next().expect("one source class");
                    if !source_dominates(lin, src, block_idx) || is_placed_value(extraction, src) {
                        survivors.push(pidx);
                        continue;
                    }
                    if src != own_class {
                        merges.push((own_class, src));
                    }
                    removed += 1;
                    dropped_any = true;
                }
                _ => survivors.push(pidx),
            }
        }
        if dropped_any {
            keep.insert(block.id, survivors);
        }
    }

    Removal {
        keep,
        removed,
        merges,
    }
}

/// Whether the block losing a parameter can name the source class as it stands.
///
/// This is the whole of tier 1's premise, stated in the terms linearization
/// works in. If the source's emitter dominates the block, the class is already
/// in a register on every path there and removing the parameter changes nothing
/// but the copies. If it does not, linearization would emit the class again in
/// this block -- which is a *cost* decision (step 2's tier 2, cost-gated), and
/// for `LoadResult`, `CallResult`, `Param` and `BlockParam` not a decision at
/// all: their value is already in a register at one particular point, and
/// re-emitting the pseudo-op mints a definition no instruction writes. A
/// re-emitted `LoadResult` also becomes a second barrier instruction for one
/// effectful op, which `compile` asserts against, and that assertion is how this
/// rule was found.
///
/// The subtree matters as much as the top node -- a source whose emitter
/// dominates has its whole subtree emitted there too -- which is why this asks
/// about emission rather than about the op.
fn is_placed_value(extraction: &ExtractionResult, class: ClassId) -> bool {
    match extraction.choices.get(&class) {
        Some(node) => matches!(
            node.op,
            Op::LoadResult(..) | Op::CallResult(..) | Op::Param(..) | Op::BlockParam(..)
        ),
        None => true,
    }
}

fn source_dominates(lin: &Linearized, src: ClassId, block_idx: usize) -> bool {
    match lin.class_emitted_in.get(&src) {
        Some(&emitter) => dominates(emitter, block_idx, &lin.idom),
        // Emitted nowhere: nothing to name.
        None => false,
    }
}

/// Drop the parameters `removal` names, and the arguments that fed them.
///
/// Unions each removed parameter's class with the one its predecessors pass
/// first: the block's own uses still name the parameter, and dropping its node
/// without the union leaves a class with nothing to extract. Then removes the
/// positions from the CFG and from the `BlockParam` nodes.
///
/// Leaves every terminator's arguments as `Classes`: their VRegs describe the
/// linearization this removal replaces, and the caller re-runs it.
pub(super) fn apply(func: &mut Function, egraph: &mut EGraph, removal: &Removal) {
    for &(param_class, src_class) in &removal.merges {
        egraph.merge(param_class, src_class);
    }
    egraph.rebuild();
    egraph.rewrite_block_params(&removal.keep);

    // Positions to drop, by target block.
    let drop: BTreeMap<BlockId, BTreeSet<u32>> = removal
        .keep
        .iter()
        .map(|(&bid, survivors)| (bid, survivors.iter().copied().collect::<BTreeSet<u32>>()))
        .collect();

    for block in func.blocks.iter_mut() {
        if let Some(kept) = drop.get(&block.id) {
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
            args.uncommit();
            let Some(kept) = drop.get(&target) else {
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

/// Whether each block's instructions carry exactly one barrier result per
/// effectful op, which is what lowering asserts.
///
/// The removal's own acceptance test. A block that loses a parameter names the
/// merged class directly, and a *merge* is not local: unioning one parameter's
/// class with its source changes what every other list naming either class
/// means, so a `CallResult` from a block that dominates nothing can end up named
/// where it is re-emitted. Re-emitting a barrier result mints a second barrier
/// instruction for one effectful op and a definition no instruction writes.
///
/// Rather than predict that composition, the pass does the removal, linearizes,
/// and asks. A `false` here puts the CFG back exactly as it was.
pub(super) fn barrier_counts_agree(func: &Function, lin: &Linearized) -> bool {
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::op::Op;
    func.blocks.iter().enumerate().all(|(idx, block)| {
        // What the block's own effectful ops produce: one result instruction per
        // Load, and one per Call that has results.
        let owned = block.ops[..block.non_term_count()]
            .iter()
            .filter(|op| match op {
                EffectfulOp::Load { .. } => true,
                EffectfulOp::Call { results, .. } => !results.is_empty(),
                _ => false,
            })
            .count();
        let emitted = lin.block_vreg_insts[idx]
            .iter()
            .filter(|inst| matches!(inst.op, Op::LoadResult(..) | Op::CallResult(..)))
            .count();
        emitted <= owned
    })
}
