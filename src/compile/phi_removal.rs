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
use crate::ir::effectful::{BlockId, EffOperand, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};

use super::barrier::{terminator_edges, terminator_edges_mut};
use super::cfg::dominates;
use super::linearize::Linearized;

/// What one parameter position receives: the VRegs its predecessors pass, the
/// classes those carry, and how many edges supplied a value.
type Incoming = (BTreeSet<VReg>, BTreeSet<ClassId>, usize);

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
    /// The parameter classes each block contributed, so a retry can drop one
    /// block's merges without recomputing the analysis.
    pub merge_owner: BTreeMap<BlockId, Vec<ClassId>>,
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
    tier2: bool,
) -> Removal {
    let block_param_vregs = &lin.block_param_vregs;
    let depths = super::cfg::block_loop_depths(func);
    let entry = func.blocks.first().map(|b| b.id);

    // Per (target, position): the VRegs its predecessors pass, the classes they
    // carry, and how many edges supplied one.
    let mut incoming: BTreeMap<(BlockId, u32), Incoming> = BTreeMap::new();
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
    let mut merge_owner: BTreeMap<BlockId, Vec<ClassId>> = BTreeMap::new();
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
            // One class naming it, whatever the registers. More than one class
            // is a real phi: predecessors computed different expressions, and
            // nothing can stand in for the parameter.
            let (Some(own_class), 1) = (own_class, source_classes.len()) else {
                survivors.push(pidx);
                continue;
            };
            let src = *source_classes.iter().next().expect("one source class");
            // A value already in a register at one particular point -- put there
            // by the caller, a load, a call, or a predecessor's phi copy -- has
            // no re-emission available, so no tier may take it.
            //
            // Vetoing these even where the source's emitter *dominates*, which
            // needs no re-emission, is stricter than soundness requires. It was
            // measured: allowing them took seed 22's removals from 13 to 40 and
            // cost pressure seed 19 at -O0, which stopped compiling. The
            // parameter was holding a live range apart that the allocator then
            // could not colour. Worth revisiting with the splitter, not before.
            if is_placed_value(extraction, src) {
                survivors.push(pidx);
                continue;
            }
            let dominating = source_dominates(lin, src, block_idx);
            let keep_it = if dominating {
                // Tier 1: the value is in one register on every path into the
                // block and nothing is re-emitted, so only the copies go.
                false
            } else {
                // Tier 2: several registers holding one expression, which is
                // what the parameter exists for. Removing it makes the block
                // recompute, so the dial decides.
                !tier2_pays(extraction, src, *edges, depths[block_idx], tier2)
            };
            if keep_it {
                survivors.push(pidx);
                continue;
            }
            if src != own_class {
                merges.push((own_class, src));
                merge_owner.entry(block.id).or_default().push(own_class);
            }
            removed += 1;
            dropped_any = true;
        }
        if dropped_any {
            keep.insert(block.id, survivors);
        }
    }

    Removal {
        keep,
        removed,
        merges,
        merge_owner,
    }
}

/// Whether recomputing the class in the block beats the copies it removes.
///
/// **The dial, with a derived default.** What removal costs is one extraction of
/// the class in this block. What it saves is one copy per incoming edge, plus
/// the parameter's slot in the clique every other parameter of the block must be
/// coloured against. Both sides are weighted by the block's loop depth, the same
/// way the splitter weights a spill: a copy inside a loop is paid every
/// iteration and so is a recomputation.
///
/// The weight is `2^depth` capped at 6 doublings, which is `compute_loop_depths`'
/// own heuristic read as a frequency. It cancels between the two sides -- both
/// happen in the same block -- so it is written here only to say that it was
/// considered and why it does not appear.
fn tier2_pays(
    extraction: &ExtractionResult,
    src: ClassId,
    edges: usize,
    _depth: u32,
    enabled: bool,
) -> bool {
    if !enabled {
        return false;
    }
    let Some(node) = extraction.choices.get(&src) else {
        return false;
    };
    // One copy per edge, plus the clique slot the parameter occupies.
    let saved = edges as f64 + 1.0;
    node.cost <= saved
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
        // Every VReg the CFG holds describes a linearization this removal is
        // about to replace, so none of them outlives the removal: the classes
        // are still what the operands are, and the VRegs are an answer the next
        // linearization re-asks.
        for op in block.ops.iter_mut() {
            match op {
                EffectfulOp::Load { addr, result, .. } => {
                    addr.uncommit();
                    result.uncommit();
                }
                EffectfulOp::Store { addr, val, .. } => {
                    addr.uncommit();
                    val.uncommit();
                }
                EffectfulOp::Call { args, results, .. } => args
                    .iter_mut()
                    .chain(results.iter_mut())
                    .for_each(EffOperand::uncommit),
                EffectfulOp::Branch { cond, .. } => cond.uncommit(),
                _ => {}
            }
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
pub(super) fn barrier_offenders(func: &Function, lin: &Linearized) -> BTreeSet<BlockId> {
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::op::Op;
    let mut offenders = BTreeSet::new();
    for (idx, block) in func.blocks.iter().enumerate() {
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
        if emitted > owned {
            offenders.insert(block.id);
        }
    }
    offenders
}

impl Removal {
    /// Drop the removals for `blocks`, keeping the rest.
    ///
    /// The acceptance test names the block that would re-emit a barrier result,
    /// and that block is the one whose parameter should not have gone. Retrying
    /// without it keeps what the rest of the function had earned: on a
    /// pressure-shaped program the all-or-nothing version put back all thirteen
    /// removals because of one.
    pub fn without(&self, func: &Function, blocks: &BTreeSet<BlockId>) -> Removal {
        let mut keep = self.keep.clone();
        let mut removed = 0usize;
        for (&bid, survivors) in &self.keep {
            if blocks.contains(&bid) {
                keep.remove(&bid);
            } else if let Some(block) = func.blocks.iter().find(|b| b.id == bid) {
                removed += block.param_types.len() - survivors.len();
            }
        }
        // A merge belongs to the block whose parameter it replaced; without the
        // per-block record, keep only merges for blocks still losing something.
        let kept_classes: BTreeSet<ClassId> = keep
            .keys()
            .filter_map(|bid| self.merge_owner.get(bid))
            .flatten()
            .copied()
            .collect();
        let merges = self
            .merges
            .iter()
            .filter(|(param_class, _)| kept_classes.contains(param_class))
            .copied()
            .collect();
        Removal {
            keep,
            removed,
            merges,
            merge_owner: self.merge_owner.clone(),
        }
    }
}
