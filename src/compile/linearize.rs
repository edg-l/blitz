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

use super::cfg::{self, collect_block_roots, compute_idom, compute_rpo, dominates};
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
    let mut class_emitted_in: BTreeMap<ClassId, usize> = BTreeMap::new();

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

    // Build per-block VRegInst lists in RPO order, stored by block index.
    let mut block_vreg_insts: Vec<Vec<VRegInst>> = vec![Vec::new(); func.blocks.len()];
    // Snapshot of class_to_vreg AT THE END of each block's processing (before
    // `removed` is restored). Captures the block-local view: classes re-emitted
    // in a block point to that block's VReg, not the globally-restored one.
    let mut block_class_to_vreg_snapshot: Vec<ClassVRegMap> =
        vec![ClassVRegMap::new(); func.blocks.len()];
    for &block_idx in &rpo_order {
        // Remove classes emitted in non-dominating blocks so they get fresh VRegs.
        // Also remove from ALL prior blocks every class whose value lives in a
        // fixed physical register rather than in the VReg it is assigned, since
        // no such register survives a block boundary.
        let removable_classes: Vec<ClassId> = class_emitted_in
            .iter()
            .filter(|(cid, emitter)| {
                if !dominates(**emitter, block_idx, &idom) {
                    return true;
                }
                if **emitter != block_idx {
                    // EFLAGS: any arithmetic instruction clobbers them.
                    let ty = &egraph.classes[cid.0 as usize].ty;
                    if matches!(ty, Type::Flags)
                        || matches!(ty, Type::Pair(_, b) if **b == Type::Flags)
                    {
                        return true;
                    }
                    if div_classes.contains(cid) {
                        return true;
                    }
                }
                false
            })
            .map(|(cid, _)| *cid)
            .collect();
        let mut removed: Vec<(ClassId, VReg)> = Vec::new();
        for cid in removable_classes {
            if let Some(vreg) = class_to_vreg.remove(cid) {
                removed.push((cid, vreg));
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
        let pre_emission: BTreeSet<ClassId> = class_to_vreg.keys().collect();
        let mut insts =
            vreg_insts_for_block(extraction, &all_roots, &mut class_to_vreg, &mut next_vreg);

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
                    } else if pre_emission.contains(&canon) && block_preds[block_idx].len() <= 1 {
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

        // Track newly emitted classes for dominator filtering.
        for cid in class_to_vreg.keys().collect::<Vec<_>>() {
            if !pre_emission.contains(&cid) && !class_emitted_in.contains_key(&cid) {
                class_emitted_in.insert(cid, block_idx);
            }
        }

        // Snapshot class_to_vreg BEFORE restore: this is the block-local view.
        // Later block lowering uses this so classes re-emitted in a block
        // resolve to that block's VReg, not a stale cross-block one.
        block_class_to_vreg_snapshot[block_idx] = class_to_vreg.clone();

        // Restore removed classes so subsequent blocks can see them.
        for (cid, vreg) in removed {
            class_to_vreg.insert_single(cid, vreg);
        }
    }

    // Build VReg -> Type map from the egraph's per-class type info.
    //
    // From the per-block snapshots as well as the function-wide map, because a
    // class re-emitted in a later block gets a VReg of its own and the restore
    // above is an `insert_single` -- it replaces the class's segments, so the
    // function-wide map keeps one re-emission and every other one is left with no
    // type at all. Lowering derives an operand width from this map and a missing
    // entry falls back to 64 bits, which is a miscompile rather than a pessimism:
    // a flags-only `cmp` on two I32 values came out `cmp r8,rdi`, and `mov edi,-2`
    // had zero-extended, so `14 < -2` compared 14 against 4294967294 and was true.
    let mut vreg_types = build_vreg_types(&class_to_vreg, egraph);
    for snapshot in &block_class_to_vreg_snapshot {
        vreg_types.extend(build_vreg_types(snapshot, egraph));
    }

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
