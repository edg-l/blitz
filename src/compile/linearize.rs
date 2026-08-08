//! Linearization: the CFG's pure values become per-block `VRegInst` lists.
//!
//! One e-class can end up with several VRegs, and this is where that is decided:
//! a class is emitted once in the first block that reaches it, and re-emitted in
//! any block its emitter does not dominate. Everything downstream that asks
//! "which register carries this class here" is asking about a choice made here,
//! which is why the answer is returned as per-block snapshots rather than one
//! function-wide map.
//!
//! Runs a second time when [`super::phi_removal`] finds parameters to remove:
//! the removal re-runs this rather than patching its output, because the block
//! that loses a parameter needs the class re-emitted in it, and this is the pass
//! that decides re-emission.

use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::EGraph;
use crate::egraph::extract::{
    ClassVRegMap, ExtractionResult, VReg, VRegInst, build_vreg_types, vreg_insts_for_block,
};
use crate::ir::effectful::BlockId;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, MachOp, Op, PureOp};
use crate::ir::types::Type;

use super::cfg::{self, DomOrder, collect_block_roots, compute_idom, compute_rpo};
use super::licm::ExtraRoots;
use super::program_point::ProgramPoint;

/// What linearization decided, for the passes that run on its output.
pub(super) struct Linearized {
    /// Function-wide class -> VReg map, holding the last emission of each class.
    pub class_to_vreg: ClassVRegMap,
    /// The next unused VReg index.
    pub next_vreg: u32,
    /// Block indices in reverse post-order.
    pub rpo_order: Vec<usize>,
    /// The instructions each block emits, by block index.
    pub block_vreg_insts: Vec<Vec<VRegInst>>,
    /// `class_to_vreg` as each block saw it, captured before the restore. A
    /// class re-emitted in a block resolves to *that* block's VReg here.
    pub block_snapshots: Vec<ClassVRegMap>,
    /// Every block param's VReg, fresh or reused.
    pub block_param_vregs: BTreeMap<(BlockId, u32), VReg>,
    /// VReg -> type, from the snapshots as well as the function-wide map.
    pub vreg_types: BTreeMap<VReg, Type>,
    /// The block index that first emitted each class. A block whose emitter does
    /// not dominate it gets its own copy, so this is the record of which classes
    /// a later pass may name without forcing a re-emission.
    pub class_emitted_in: BTreeMap<ClassId, usize>,
    /// Immediate dominator per block index, for asking that question.
    pub idom: Vec<Option<usize>>,
}

/// Build the per-block `VRegInst` lists and everything derived from that choice.
pub(super) fn linearize(
    func: &Function,
    egraph: &EGraph,
    extraction: &ExtractionResult,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    extra_roots: &ExtraRoots,
) -> Linearized {
    // Phase 3: Build per-block VRegInst lists with a shared class_to_vreg map.
    //
    // We process blocks in RPO order so that loop headers come before loop
    // bodies and dominant definitions are visited before their uses.
    // Classes shared between blocks are only emitted by the first block that
    // reaches them (DFS deduplication).
    // DO NOT pre-populate class_to_vreg here — let the DFS assign VRegs
    // naturally so that param/block-param VRegInsts appear in the scheduled
    // list and regalloc can see them.
    let mut class_to_vreg = ClassVRegMap::new();
    let mut next_vreg: u32 = 0;

    // Compute RPO block ordering (indices into func.blocks).
    let rpo_order = compute_rpo(func);
    // Predecessor counts per block index. Used by block_param_fixup to
    // distinguish loop headers / merge points (multi-pred, need phi storage)
    // from pass-through blocks (single-pred, the block param IS its sole
    // predecessor's argument and doesn't need a fresh VReg).
    let block_preds = cfg::predecessor_indices(func);

    // Every block param's VReg, recorded as linearization decides it -- both the
    // fresh ones above and the ones that reuse an inst already in the block.
    //
    // `build_phi_copies` needs the destination of each phi copy, and re-deriving
    // it from `class_to_vreg` at the target's entry does not survive the
    // splitter: a cross-block slot spill truncates the class's segment to the
    // defining block, and every later block that carries the class as a param
    // then resolves to nothing ("param class ... not in class_to_vreg", 9 of 40
    // generated programs at -O1). The decision is made here and only here, so
    // it is recorded here rather than reconstructed downstream.
    let mut block_param_vregs: BTreeMap<(BlockId, u32), VReg> = BTreeMap::new();

    let idom = compute_idom(func, &rpo_order);
    let dom = DomOrder::new(&idom);
    let mut class_emitted_in: BTreeMap<ClassId, usize> = BTreeMap::new();

    // The same record as `class_emitted_in`, indexed by the emitting block.
    //
    // Which classes a block must re-emit is a question about their *emitters*:
    // one that does not dominate this block, or a value that lives in a fixed
    // physical register no block boundary preserves. Asked of every class in
    // turn it costs the whole function at every block; asked of every emitter it
    // costs one dominance test per block that has emitted anything, and the
    // classes come along in a list. `reemit_per_block` is the sublist that has
    // to be re-emitted even under a dominating emitter, so the common case
    // touches no class at all.
    let mut emitted_by: Vec<Vec<ClassId>> = vec![Vec::new(); func.blocks.len()];
    let mut reemit_per_block: Vec<Vec<ClassId>> = vec![Vec::new(); func.blocks.len()];
    let mut emitter_blocks: Vec<usize> = Vec::new();
    // Emitter blocks whose classes are out of the map right now, and the VReg
    // each of those classes had when it was taken out.
    //
    // A block re-emits a class whose emitter does not dominate it, and the map
    // is what tells the emission walk that. Taking every such class out and
    // putting it back around each block is the same answer recomputed from
    // nothing 3763 times: 29.7 million removals on one function, against 2419
    // changes in *which emitters* are out of scope. So the map is carried from
    // block to block and only the difference is paid -- a bucket leaves when
    // its emitter stops dominating and comes back, with the VRegs it had, when
    // it starts again.
    let mut emitter_removed: Vec<bool> = vec![false; func.blocks.len()];
    let mut removed_vregs: BTreeMap<ClassId, VReg> = BTreeMap::new();
    // Emitter blocks with at least one class that must be re-emitted even under
    // a dominating emitter. Those depend on which block is being processed, so
    // they are taken out and put back per block -- there are very few.
    let mut reemit_emitters: Vec<usize> = Vec::new();

    // A division leaves its quotient in RAX and its remainder in RDX; the pair
    // VReg holds neither. Only the projections adjacent to the division read
    // those registers, so a projection in another block would take whatever that
    // block last put there -- these classes are re-emitted per block, as flags
    // are and for the same reason. Collected once: the test below runs for every
    // (block, emitted class) pair, where a lookup in the whole extraction costs
    // more than the whole set of divisions.
    let div_classes: BTreeSet<ClassId> = extraction
        .choices
        .iter()
        .filter(|(_, node)| matches!(node.op, Op::Mach(MachOp::X86Idiv(..) | MachOp::X86Div(..))))
        .map(|(cid, _)| *cid)
        .collect();

    // Where each class is emitted, when that is not the first block to name it.
    //
    // **A class is one expression, and it should be computed once in a block
    // that reaches every use.** Emitting it in the first block that happens to
    // name it puts it in the wrong place whenever two siblings both use it:
    // neither dominates the other, so the second re-emits, and a diamond whose
    // arms share a computation pays for it twice. The e-graph already made the
    // two arms one class -- cross-block CSE is not the missing part -- so what
    // is left is purely *placement*.
    //
    // The entry block's parameters were already fixed this way, one op at a
    // time: "a param that two sibling branches both read gets re-emitted in
    // each ... emitting at entry, which dominates everything, leaves exactly
    // one VReg per parameter." This is that rule for every class, with the
    // nearest common dominator in place of the entry block.
    //
    // Not hoisted: anything already re-emitted per block on purpose. Flags do
    // not survive a block boundary and a division's results live in RAX and
    // RDX, so those classes are placed where they are used and stay there.
    let placement = class_placement(
        func,
        &rpo_order,
        &idom,
        extraction,
        egraph,
        block_param_map,
        extra_roots,
        &div_classes,
    );

    // Build per-block VRegInst lists in RPO order, stored by block index.
    let mut block_vreg_insts: Vec<Vec<VRegInst>> = vec![Vec::new(); func.blocks.len()];
    // Snapshot of class_to_vreg AT THE END of each block's processing (before
    // `removed` is restored). Captures the block-local view: classes re-emitted
    // in a block point to that block's VReg, not the globally-restored one.
    let mut block_class_to_vreg_snapshot: Vec<ClassVRegMap> =
        vec![ClassVRegMap::new(); func.blocks.len()];
    // VReg -> type, accumulated as each VReg is minted.
    let mut vreg_types: BTreeMap<VReg, Type> = BTreeMap::new();
    for &block_idx in &rpo_order {
        // Bring the scope to this block: classes emitted in non-dominating
        // blocks are out of the map, so they get fresh VRegs here, and classes
        // whose value lives in a fixed physical register are out of it too,
        // since no such register survives a block boundary.
        for &emitter in &emitter_blocks {
            let out_of_scope = !dom.dominates(emitter, block_idx);
            if out_of_scope == emitter_removed[emitter] {
                continue;
            }
            emitter_removed[emitter] = out_of_scope;
            if out_of_scope {
                for &cid in &emitted_by[emitter] {
                    if let Some(vreg) = class_to_vreg.remove(cid) {
                        removed_vregs.insert(cid, vreg);
                    }
                }
            } else {
                for &cid in &emitted_by[emitter] {
                    if let Some(vreg) = removed_vregs.remove(&cid) {
                        class_to_vreg.insert_single(cid, vreg);
                    }
                }
            }
        }

        // Flags are clobbered by any arithmetic instruction and a division's
        // pair lives in RAX and RDX, so those are re-emitted in every block that
        // names them, dominating emitter or not.
        let mut reemit_removed: Vec<(ClassId, VReg)> = Vec::new();
        for &emitter in &reemit_emitters {
            if emitter == block_idx || emitter_removed[emitter] {
                continue;
            }
            for &cid in &reemit_per_block[emitter] {
                if let Some(vreg) = class_to_vreg.remove(cid) {
                    reemit_removed.push((cid, vreg));
                }
            }
        }

        let block = &func.blocks[block_idx];
        let roots = collect_block_roots(block, egraph);
        // Also include the block param ClassIds as roots for this block so
        // they get VRegs assigned (even though BlockParam emits no instructions).
        let block_id = block.id;
        let mut all_roots = roots;
        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block_id, pidx)) {
                all_roots.push(cid);
            }
        }
        // Include LICM-hoisted roots for this block (invariant classes to emit here).
        if let Some(hoisted) = extra_roots.get(&block_idx) {
            all_roots.extend(hoisted.iter().copied());
        }
        // Classes this block is the nearest common dominator of the users of.
        if let Some(placed) = placement.get(&block_idx) {
            all_roots.extend(placed.iter().copied());
        }
        // Every parameter is a root of the entry block, whether or not the entry
        // block uses it.
        //
        // A Param op names a value the ABI already placed in a register; it
        // computes nothing. Emitted lazily in the first block that happens to
        // use it, a param that two sibling branches both read gets re-emitted in
        // each -- neither emitter dominates the other -- and only one of the two
        // VRegs carries the ABI precolor. The other is free to land anywhere, so
        // `movsd xmm0,xmm1` would "copy" a parameter out of a register that
        // never held it. Emitting at entry, which dominates everything, leaves
        // exactly one VReg per parameter.
        if block_idx == 0 {
            all_roots.extend(
                func.param_class_ids
                    .iter()
                    .map(|&cid| egraph.unionfind.find_immutable(cid)),
            );
        }
        all_roots.sort_by_key(|c| c.0);
        all_roots.dedup();
        // Whether a class was already in the map when this block started. Every
        // class the map holds is one some block emitted, and what is out of
        // scope right now is exactly what the two removal records name, so
        // between them they answer it -- `class_emitted_in` does not learn about
        // this block until the end of the iteration.
        let reemit_removed_set: BTreeSet<ClassId> =
            reemit_removed.iter().map(|(cid, _)| *cid).collect();
        let was_pre_emitted = |cid: ClassId| {
            class_emitted_in.contains_key(&cid)
                && !removed_vregs.contains_key(&cid)
                && !reemit_removed_set.contains(&cid)
        };

        let (mut insts, newly_emitted) =
            vreg_insts_for_block(extraction, &all_roots, &mut class_to_vreg, &mut next_vreg);

        // Every VReg this block minted, recorded with its class's type here and
        // not later: a re-emission's VReg is about to be replaced in the map by
        // the one its class had before -- the restore below, and for a flags
        // class every block does it -- after which only this block's snapshot
        // still names it. Reading the types back out of the snapshots afterwards
        // means reading the whole function once per block.
        for &cid in &newly_emitted {
            if let Some(vreg) = class_to_vreg.lookup_any(cid) {
                let canon = egraph.unionfind.find_immutable(cid);
                vreg_types.insert(vreg, egraph.class(canon).ty.clone());
            }
        }

        // Per-block fixup: ensure block params of this block use Op::BlockParam,
        // not whatever the global extraction chose. The global extraction picks
        // one op per e-class, but BlockParam is only meaningful in its own block.
        // Only fix up VRegInsts that were emitted in THIS block (not ones from
        // prior blocks -- cross-block splitting handles those via spill/reload).
        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block_id, pidx)) {
                let canon = egraph.unionfind.find_immutable(cid);
                if let Some(vreg) =
                    class_to_vreg.lookup(canon, ProgramPoint::block_entry(block_idx))
                {
                    if let Some(inst) = insts.iter_mut().find(|i| i.dst == vreg) {
                        inst.op = Op::Pure(PureOp::BlockParam(
                            block_id,
                            pidx,
                            block.param_types[pidx as usize].clone(),
                        ));
                        inst.operands.clear();
                        block_param_vregs.insert((block_id, pidx), vreg);
                    } else if was_pre_emitted(canon) && block_preds[block_idx].len() <= 1 {
                        // Pass-through: the canonical class was already emitted
                        // in a dominating block (survived this block's filter)
                        // AND this block has at most one predecessor, so
                        // propagate_block_params merged the param with the
                        // dominating definition and no phi storage is needed.
                        // Skipping prevents creating a dead BlockParam VReg
                        // that the regalloc places in a caller-saved register,
                        // only to be clobbered by a subsequent call in this
                        // block — later users (including Ret) would then find
                        // the dead VReg instead of the live dominating one.
                        //
                        // Multi-predecessor blocks (loop headers, merge points)
                        // still need the else branch: each predecessor passes
                        // a distinct value via phi copy into a shared storage
                        // slot, so a fresh VReg local to this block is
                        // required.
                        //
                        // The param still needs a recorded VReg even though it
                        // gets no BlockParam of its own: it is the dominating
                        // definition's, and once the splitter truncates that
                        // VReg's segment to its defining block, nothing else
                        // can name it here.
                        block_param_vregs.insert((block_id, pidx), vreg);
                        continue;
                    } else {
                        // The VReg was emitted by a non-dominating prior block.
                        // Allocate a fresh VReg local to this block to avoid
                        // outer/inner loop header param aliasing.
                        let fresh_vreg = VReg(next_vreg);
                        next_vreg += 1;
                        // Rewrite all operand references to the old vreg in
                        // this block's insts to use the fresh VReg.
                        for inst in insts.iter_mut() {
                            for operand in inst.operands.iter_mut() {
                                if *operand == Some(vreg) {
                                    *operand = Some(fresh_vreg);
                                }
                            }
                        }
                        // Add a BlockParam instruction for the fresh VReg.
                        insts.push(VRegInst {
                            dst: fresh_vreg,
                            op: Op::Pure(PureOp::BlockParam(
                                block_id,
                                pidx,
                                block.param_types[pidx as usize].clone(),
                            )),
                            operands: vec![],
                        });
                        block_param_vregs.insert((block_id, pidx), fresh_vreg);
                    }
                }
            }
        }

        block_vreg_insts[block_idx] = insts;

        // Track newly emitted classes for dominator filtering. `vreg_insts_for_block`
        // returns exactly the classes it added, so this costs what the block
        // emitted rather than what the function has emitted so far.
        for &cid in &newly_emitted {
            // The *first* block to emit a class owns it. A later block that
            // re-emits it gets a VReg of its own, but the record of where the
            // class came from must not move, or every dominance question asked
            // about it afterwards is asked about the wrong block.
            if let std::collections::btree_map::Entry::Vacant(slot) = class_emitted_in.entry(cid) {
                slot.insert(block_idx);
                if emitted_by[block_idx].is_empty() && reemit_per_block[block_idx].is_empty() {
                    emitter_blocks.push(block_idx);
                }
                emitted_by[block_idx].push(cid);
                // EFLAGS: any arithmetic instruction clobbers them. A division's
                // pair lives in RAX and RDX, which no block boundary preserves.
                let ty = &egraph.classes[cid.0 as usize].ty;
                if matches!(ty, Type::Flags)
                    || matches!(ty, Type::Pair(_, b) if **b == Type::Flags)
                    || div_classes.contains(&cid)
                {
                    if reemit_per_block[block_idx].is_empty() {
                        reemit_emitters.push(block_idx);
                    }
                    reemit_per_block[block_idx].push(cid);
                }
            }
        }

        // Snapshot class_to_vreg BEFORE restore: this is the block-local view.
        // Later block lowering uses this so classes re-emitted in a block
        // resolve to that block's VReg, not a stale cross-block one.
        //
        // Only the block's own roots: every question ever asked of a snapshot is
        // "which VReg carries a class this block's effectful ops or terminator
        // name", and those classes are exactly `all_roots`. The children the
        // emission walk gave VRegs to along the way are named by the schedule,
        // never by the CFG, so copying the whole map here would be a copy of the
        // function per block.
        let mut snapshot = ClassVRegMap::new();
        for &cid in &all_roots {
            if let Some(vreg) = class_to_vreg.lookup_any(cid) {
                snapshot.insert_full_range_shared(cid, vreg);
            }
        }
        block_class_to_vreg_snapshot[block_idx] = snapshot;

        // Put back the per-block re-emitted classes, and take the block's own
        // copy of an out-of-scope class back out. The scope carries over to the
        // next block, so a class whose emitter still does not dominate must not
        // be left in the map holding the VReg this block minted for it -- the
        // one it had when it went out of scope is the one waiting to come back.
        for (cid, vreg) in reemit_removed {
            class_to_vreg.insert_single(cid, vreg);
        }
        for &cid in &newly_emitted {
            if removed_vregs.contains_key(&cid) {
                class_to_vreg.remove(cid);
            }
        }
    }

    // Every class the scope still holds out belongs in the function-wide map the
    // later passes read, which is the whole of it rather than any block's view.
    for (cid, vreg) in std::mem::take(&mut removed_vregs) {
        class_to_vreg.insert_single(cid, vreg);
    }

    // The function-wide map on top of what the blocks recorded as they emitted.
    //
    // Both are needed: a class re-emitted in a later block gets a VReg of its
    // own and the restore above is an `insert_single`, so the function-wide map
    // keeps one re-emission and every other one would be left with no type at
    // all. Lowering derives an operand width from this map and a missing entry
    // falls back to 64 bits, which is a miscompile rather than a pessimism: a
    // flags-only `cmp` on two I32 values came out `cmp r8,rdi`, and `mov edi,-2`
    // had zero-extended, so `14 < -2` compared 14 against 4294967294 and was
    // true.
    vreg_types.extend(build_vreg_types(&class_to_vreg, egraph));

    // Every block parameter's VReg needs a type, including one linearization
    // minted fresh above: it has no e-class entry of its own, and a VReg absent
    // from `vreg_types` makes lowering fall back to 64 bits.
    let block_id_to_idx = cfg::block_id_to_idx(func);
    for (&(bid, pidx), &vreg) in &block_param_vregs {
        let block = &func.blocks[block_id_to_idx[&bid]];
        let ty = block.param_types[pidx as usize].clone();
        vreg_types.insert(vreg, ty);
    }

    Linearized {
        class_to_vreg,
        next_vreg,
        rpo_order,
        block_vreg_insts,
        block_snapshots: block_class_to_vreg_snapshot,
        block_param_vregs,
        vreg_types,
        class_emitted_in,
        idom,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// A subtree cheaper than this is re-emitted where it is used rather than
/// placed above its users: one spill store and one reload is what hoisting
/// risks, and the splitter prices that pair at 5.0.
const PLACEMENT_MIN_COST: f64 = 5.0;

/// The block each class is emitted in: the nearest common dominator of every
/// block that needs it.
///
/// Returns only the classes whose placement differs from the block that would
/// otherwise emit them, keyed by that block, so the emission walk adds them to
/// its roots and everything else is untouched.
///
/// **Needing a class is transitive.** A block that names a root needs the whole
/// subtree the extraction chose for it, so the walk below closes over
/// `children`. That also makes the result consistent by construction: a child is
/// needed everywhere its parent is, so the child's nearest common dominator
/// dominates the parent's, and every operand is in scope where its consumer is
/// emitted.
///
/// A class placed above every block that uses it is live from there to its last
/// use, where re-emitting kept each copy's range short. That is the same trade
/// rematerialization makes in the other direction, and it is why this is judged
/// on measurement rather than on being obviously right.
#[allow(clippy::too_many_arguments)]
fn class_placement(
    func: &Function,
    rpo_order: &[usize],
    idom: &[Option<usize>],
    extraction: &ExtractionResult,
    egraph: &EGraph,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    extra_roots: &ExtraRoots,
    div_classes: &BTreeSet<ClassId>,
) -> BTreeMap<usize, Vec<ClassId>> {
    let mut rpo_pos = vec![usize::MAX; func.blocks.len()];
    for (pos, &b) in rpo_order.iter().enumerate() {
        rpo_pos[b] = pos;
    }
    // The nearest block that dominates both, walking the two idom chains
    // towards the entry. `None` where either block is unreachable and so in no
    // dominator tree.
    let ncd = |mut a: usize, mut b: usize| -> Option<usize> {
        while a != b {
            if rpo_pos[a] == usize::MAX || rpo_pos[b] == usize::MAX {
                return None;
            }
            if rpo_pos[a] > rpo_pos[b] {
                a = idom[a]?;
            } else {
                b = idom[b]?;
            }
        }
        Some(a)
    };

    // The first block to name each class, and where it must end up.
    let mut first_seen: BTreeMap<ClassId, usize> = BTreeMap::new();
    let mut place: BTreeMap<ClassId, Option<usize>> = BTreeMap::new();
    let mut stack: Vec<ClassId> = Vec::new();
    let mut seen_here: BTreeSet<ClassId> = BTreeSet::new();

    for &block_idx in rpo_order {
        let block = &func.blocks[block_idx];
        let mut roots = cfg::collect_block_roots(block, egraph);
        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block.id, pidx)) {
                roots.push(cid);
            }
        }
        if let Some(hoisted) = extra_roots.get(&block_idx) {
            roots.extend(hoisted.iter().copied());
        }
        if block_idx == 0 {
            roots.extend(
                func.param_class_ids
                    .iter()
                    .map(|&cid| egraph.unionfind.find_immutable(cid)),
            );
        }

        seen_here.clear();
        stack.extend(roots);
        while let Some(cid) = stack.pop() {
            if !seen_here.insert(cid) {
                continue;
            }
            first_seen.entry(cid).or_insert(block_idx);
            let slot = place.entry(cid).or_insert(Some(block_idx));
            if let Some(cur) = *slot {
                *slot = ncd(cur, block_idx);
            }
            if let Some(node) = extraction.choices.get(&cid) {
                stack.extend(
                    node.children
                        .iter()
                        .copied()
                        .filter(|c| *c != ClassId::NONE),
                );
            }
        }
    }

    let mut out: BTreeMap<usize, Vec<ClassId>> = BTreeMap::new();
    for (cid, target) in place {
        let Some(target) = target else { continue };
        if first_seen.get(&cid) == Some(&target) {
            continue; // already emitted where it belongs
        }
        // A class re-emitted per block on purpose is not placed.
        if div_classes.contains(&cid) {
            continue;
        }
        let ty = &egraph.classes[cid.0 as usize].ty;
        if matches!(ty, Type::Flags) || matches!(ty, Type::Pair(_, b) if **b == Type::Flags) {
            continue;
        }
        // A block parameter is the block's own, and means nothing anywhere else.
        if matches!(
            extraction.choices.get(&cid).map(|n| &n.op),
            Some(Op::Pure(PureOp::BlockParam(..)))
        ) {
            continue;
        }
        // Only where recomputing costs more than keeping the value live.
        //
        // Placing a class above its users is a trade, not an improvement:
        // computed once, it is then live from the dominator to the last use and
        // the allocator may have to spill it, where re-emitting kept each range
        // inside one block. Unrestricted, the trade loses -- `+6.5%` spills,
        // `+10.1%` reloads, `+13.8%` cycles -- so it is taken only for a subtree
        // dear enough that one store and one reload is the cheaper end of it.
        // The threshold is the splitter's, which prices exactly that pair.
        if extraction
            .choices
            .get(&cid)
            .is_none_or(|n| n.cost <= PLACEMENT_MIN_COST)
        {
            continue;
        }
        out.entry(target).or_default().push(cid);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::enode::ENode;
    use crate::egraph::extract::{ExtractedNode, VRegInst};
    use crate::ir::CondCode;
    use crate::ir::effectful::{EffOperand, EffectfulOp, TermArgs};
    use crate::ir::function::BasicBlock;

    /// An e-graph of `n` distinct I64 constants, class `i` holding `iconst(i)`.
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

    /// The only extraction an e-graph of constants has: each class's one node,
    /// with every subtree priced at `cost` -- which is the one input placement
    /// reads, since it decides only whether recomputing is dearer than keeping
    /// the value live.
    fn extraction_costing(egraph: &EGraph, cost: f64) -> ExtractionResult {
        ExtractionResult {
            choices: (0..egraph.arena_len() as u32)
                .map(|i| {
                    let class = egraph.class(ClassId(i));
                    (
                        ClassId(i),
                        ExtractedNode {
                            op: class.nodes[0].op.clone(),
                            children: vec![],
                            cost,
                        },
                    )
                })
                .collect(),
        }
    }

    /// Blocks 0..n, block `i` carrying `ops[i]`, ids equal to indices.
    fn func_with(ops: Vec<Vec<EffectfulOp>>, param_class_ids: Vec<ClassId>) -> Function {
        let blocks = ops
            .into_iter()
            .enumerate()
            .map(|(i, ops)| BasicBlock {
                id: i as BlockId,
                param_types: vec![],
                param_vregs: vec![],
                ops,
            })
            .collect::<Vec<_>>();
        Function {
            name: "f".to_string(),
            param_types: vec![Type::I64; param_class_ids.len()],
            return_types: vec![],
            next_block_id: blocks.len() as BlockId,
            blocks,
            param_class_ids,
            egraph: None,
            stack_slots: vec![],
            noinline: false,
        }
    }

    /// A store naming `class` as both its address and its value, so a block
    /// carrying one names exactly that class and nothing else.
    fn store(class: ClassId) -> EffectfulOp {
        EffectfulOp::Store {
            addr: EffOperand::Class(class),
            ty: Type::I64,
            val: EffOperand::Class(class),
        }
    }

    fn branch(cond: ClassId, bb_true: BlockId, bb_false: BlockId) -> EffectfulOp {
        EffectfulOp::Branch {
            cond: EffOperand::Class(cond),
            cc: CondCode::Ne,
            bb_true,
            bb_false,
            true_args: TermArgs::classes([]),
            false_args: TermArgs::classes([]),
        }
    }

    fn jump(target: BlockId) -> EffectfulOp {
        EffectfulOp::Jump {
            target,
            args: TermArgs::classes([]),
        }
    }

    fn ret() -> EffectfulOp {
        EffectfulOp::Ret { val: None }
    }

    fn run(func: &Function, egraph: &EGraph) -> Linearized {
        run_costing(func, egraph, 0.0)
    }

    fn run_costing(func: &Function, egraph: &EGraph, cost: f64) -> Linearized {
        linearize(
            func,
            egraph,
            &extraction_costing(egraph, cost),
            &BTreeMap::new(),
            &ExtraRoots::new(),
        )
    }

    /// Whether `block` emitted an instruction for `class`, asked of the block's
    /// own snapshot rather than the function-wide map.
    fn emits(lin: &Linearized, block_idx: usize, class: ClassId) -> bool {
        let Some(vreg) = lin.block_snapshots[block_idx].lookup_any(class) else {
            return false;
        };
        lin.block_vreg_insts[block_idx]
            .iter()
            .any(|i: &VRegInst| i.dst == vreg)
    }

    /// 0 branches to 1 and 2, both of which name class 1, and both jump to 3.
    fn diamond_naming_class_one() -> (Function, EGraph) {
        let func = func_with(
            vec![
                vec![branch(ClassId(0), 1, 2)],
                vec![store(ClassId(1)), jump(3)],
                vec![store(ClassId(1)), jump(3)],
                vec![ret()],
            ],
            vec![],
        );
        (func, constants(2))
    }

    /// 0 names class 1 and jumps to 1, which names it again.
    fn chain_naming_class_one() -> (Function, EGraph) {
        let func = func_with(
            vec![
                vec![store(ClassId(1)), jump(1)],
                vec![store(ClassId(1)), ret()],
            ],
            vec![],
        );
        (func, constants(2))
    }

    /// A class both arms of a diamond name, and that is dear enough to be worth
    /// keeping live, is emitted once in the block that dominates them.
    ///
    /// Neither arm dominates the other, so emitting in the first block to *name*
    /// the class makes the second re-emit and the diamond pay twice. The nearest
    /// common dominator of the two arms is block 0, which reaches both.
    #[test]
    fn a_costly_class_named_by_both_arms_is_emitted_in_their_dominator() {
        let (func, egraph) = diamond_naming_class_one();
        let lin = run_costing(&func, &egraph, PLACEMENT_MIN_COST + 1.0);

        assert_eq!(
            lin.class_emitted_in[&ClassId(1)],
            0,
            "the class belongs to the block that dominates both arms",
        );
        assert!(emits(&lin, 0, ClassId(1)));
        assert!(!emits(&lin, 1, ClassId(1)), "arm 1 must reuse it");
        assert!(!emits(&lin, 2, ClassId(1)), "arm 2 must reuse it");
        assert_eq!(
            lin.block_snapshots[1].lookup_any(ClassId(1)),
            lin.block_snapshots[2].lookup_any(ClassId(1)),
            "one emission is one VReg, which both arms read",
        );
    }

    /// A cheap one is not: recomputing it in each arm costs less than the one
    /// spill store and reload that holding it across the branch risks. Hoisting
    /// these regardless measured `+6.5%` spills and `+13.8%` cycles.
    #[test]
    fn a_cheap_class_named_by_both_arms_is_re_emitted_in_each() {
        let (func, egraph) = diamond_naming_class_one();
        let lin = run_costing(&func, &egraph, PLACEMENT_MIN_COST);

        assert!(!emits(&lin, 0, ClassId(1)), "block 0 does not name it");
        assert!(emits(&lin, 1, ClassId(1)));
        assert!(emits(&lin, 2, ClassId(1)));
        assert_ne!(
            lin.block_snapshots[1].lookup_any(ClassId(1)),
            lin.block_snapshots[2].lookup_any(ClassId(1)),
            "each arm carries its own",
        );
    }

    /// The other half of the same rule: a dominating emitter is reused, so the
    /// later block emits nothing and both snapshots name one VReg.
    #[test]
    fn a_dominating_emitter_is_not_re_emitted() {
        let (func, egraph) = chain_naming_class_one();
        let lin = run(&func, &egraph);

        assert_eq!(lin.class_emitted_in[&ClassId(1)], 0);
        assert!(emits(&lin, 0, ClassId(1)));
        assert!(!emits(&lin, 1, ClassId(1)));
        assert_eq!(
            lin.block_snapshots[0].lookup_any(ClassId(1)),
            lin.block_snapshots[1].lookup_any(ClassId(1)),
        );
    }

    /// A re-emission's VReg is replaced in the function-wide map by the one its
    /// class had before, so only its own block's snapshot still names it. Its
    /// type is recorded as it is minted for that reason, and a VReg absent from
    /// `vreg_types` makes lowering fall back to 64 bits -- a miscompile on a
    /// narrower value, not a pessimism.
    #[test]
    fn every_minted_vreg_has_a_type() {
        for (func, egraph) in [diamond_naming_class_one(), chain_naming_class_one()] {
            let lin = run(&func, &egraph);
            for (block_idx, insts) in lin.block_vreg_insts.iter().enumerate() {
                for inst in insts {
                    assert!(
                        lin.vreg_types.contains_key(&inst.dst),
                        "block {block_idx} minted v{} with no type",
                        inst.dst.0,
                    );
                }
            }
        }
    }

    /// A parameter is a root of the entry block whether or not the entry block
    /// uses it. Emitted lazily instead, a parameter two sibling branches both
    /// read is re-emitted in each -- and only one of the two VRegs carries the
    /// ABI precolor, so the other names a register that never held the value.
    #[test]
    fn a_parameter_is_emitted_once_at_entry_even_when_the_entry_does_not_use_it() {
        let func = func_with(
            vec![
                vec![branch(ClassId(0), 1, 2)],
                vec![store(ClassId(1)), jump(3)],
                vec![store(ClassId(1)), jump(3)],
                vec![ret()],
            ],
            vec![ClassId(1)],
        );
        let lin = run(&func, &constants(2));

        assert_eq!(lin.class_emitted_in[&ClassId(1)], 0);
        assert!(emits(&lin, 0, ClassId(1)));
        let entry = lin.block_snapshots[0].lookup_any(ClassId(1));
        assert!(entry.is_some());
        for arm in [1, 2] {
            assert!(
                !emits(&lin, arm, ClassId(1)),
                "block {arm} re-emitted a parameter"
            );
            assert_eq!(lin.block_snapshots[arm].lookup_any(ClassId(1)), entry);
        }
    }
}
