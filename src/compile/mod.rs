//! End-to-end compilation pipeline for Blitz.
//!
//! Phases:
//!  1. E-graph rewrite rules (algebraic, strength reduction, isel)
//!  2. Cost-based extraction
//!  3. VRegInst linearization
//!  4. DAG scheduling
//!  5. Register allocation
//!  6. VReg-to-phys rewrite
//!  7. Op -> MachInst lowering
//!  8. Peephole optimization
//!  9. NOP alignment (optional)
//! 10. Encoding
//! 11. ELF emission

use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::cost::{CostModel, OptGoal};
use crate::egraph::extract::{
    ClassVRegMap, VReg, VRegInst, build_vreg_types, extract, vreg_insts_for_block,
};
use crate::egraph::phases::{CompileOptions as EGraphOptions, run_phases};
use crate::emit::object::{FunctionInfo, ObjectFile};
use crate::emit::peephole::peephole;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::regalloc::allocate_global;
use crate::regalloc::allocator::{RegAllocResult, allocate};
use crate::regalloc::slots::SlotAllocator;
use crate::schedule::scheduler::{ScheduleDag, ScheduledInst, schedule};
use crate::x86::abi::{compute_frame_layout, emit_epilogue, emit_prologue};
use crate::x86::encode::{Encoder, inst_size};
use crate::x86::inst::{LabelId, MachInst};
use crate::x86::reg::Reg;

pub(crate) mod barrier;
pub mod program_point;
use barrier::{
    assign_barrier_groups, build_barrier_context, insert_early_barrier_spills,
    populate_effectful_operands,
};
use program_point::ProgramPoint;
mod canon;
use canon::canonicalize_class_refs;
pub(crate) mod cfg;
use cfg::{
    collect_block_roots, collect_externals, collect_phi_source_vregs, collect_roots,
    commit_terminator_arg_vregs, compute_copy_pairs, compute_copy_pairs_from_schedules,
    compute_idom, compute_loop_depths, compute_rpo, dominates,
};
mod effectful;
use effectful::lower_effectful_op;
mod dce;
mod licm;
mod lower;
mod phi_simplify;
use lower::lower_block_pure_ops;
mod precolor;
use precolor::{
    add_call_precolors_for_block, add_div_precolors, add_shift_precolors,
    assign_param_vregs_from_map,
};
mod terminator;
use terminator::{lower_terminator, thread_branches};
pub mod alias;
pub(crate) mod split;
pub use alias::{AddrBase, AliasInfo};
pub mod dse;
pub mod forward;

// ── Public options / error types ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations: fast compilation, minimal rewrites.
    O0,
    /// Full optimizations: inlining, egraph saturation, peephole.
    O1,
}

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub opt_level: OptLevel,
    pub opt_goal: OptGoal,
    /// Maximum e-graph saturation iterations. The loop exits early when no rules fire,
    /// so this is a safety cap. Typical programs converge in 2-4 iterations; 16 is generous.
    /// O0 uses 1 (minimum for isel; the lowerer requires x86 ops).
    pub saturation_limit: u32,
    pub enable_peephole: bool,
    pub enable_nop_alignment: bool,
    pub verbosity: Verbosity,
    /// Force the frame pointer (push rbp / mov rbp, rsp / pop rbp) to always be emitted.
    /// Defaults to `false`: the frame pointer is omitted when not needed, freeing RBP as a
    /// general-purpose register. Set to `true` for debuggability or when a frame pointer is
    /// required (e.g. kernel code).
    pub force_frame_pointer: bool,
    /// Enable Loop-Invariant Code Motion (LICM) before e-graph optimization.
    pub enable_licm: bool,
    /// Enable Dead Code Elimination (unreachable blocks, constant branches, dead loads).
    pub enable_dce: bool,
    /// Enable intra-block store-to-load and load-to-load forwarding.
    pub enable_store_forwarding: bool,
    /// Enable intra-block dead store elimination.
    pub enable_dse: bool,
    /// Eliminate block parameters that are trivial phis (`phi(x, ..., x) -> x`).
    ///
    /// A canonicalization rather than an optimization: SSA construction creates a
    /// parameter for every variable live across an edge and never revisits it, and
    /// 85-94% of them turn out to name a single value.
    pub enable_phi_simplify: bool,
    /// Enable function inlining before optimization.
    pub enable_inlining: bool,
    /// Maximum inlining rescan iterations per caller function. Each rescan inlines one level
    /// of calls; a depth-3 chain A->B->C->D needs 3 rescans. Note: this limits rescans,
    /// not true nesting depth; a function with many independent leaf calls also consumes
    /// iterations. Default 3 handles most practical transitive inlining.
    pub max_inline_depth: u32,
    /// Maximum callee e-graph node count (pre-saturation) for inlining eligibility.
    /// Measures raw e-nodes from IR construction, not post-optimization size. Rough proxy
    /// for code complexity. Default 50 corresponds to roughly 20-30 IR instructions.
    pub max_inline_nodes: usize,
    /// Maximum weighted cost for inlining a callee. Each IR operation has a weight
    /// (e.g. Add=1, SDiv=10, Call=20). If the total cost exceeds this threshold,
    /// the callee is not inlined (unless it has a single caller). Default 100.
    pub inline_cost_threshold: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Silent,
    Normal,
    Verbose,
}

impl CompileOptions {
    pub fn o0() -> Self {
        CompileOptions {
            opt_level: OptLevel::O0,
            opt_goal: OptGoal::Balanced,
            saturation_limit: 1,
            enable_peephole: false,
            enable_nop_alignment: false,
            verbosity: Verbosity::Silent,
            force_frame_pointer: false,
            enable_licm: false,
            enable_dce: false,
            enable_store_forwarding: false,
            enable_dse: false,
            enable_phi_simplify: false,
            enable_inlining: false,
            max_inline_depth: 3,
            max_inline_nodes: 50,
            inline_cost_threshold: 0,
        }
    }

    pub fn o1() -> Self {
        CompileOptions {
            opt_level: OptLevel::O1,
            opt_goal: OptGoal::Balanced,
            saturation_limit: 16,
            enable_peephole: true,
            enable_nop_alignment: false,
            verbosity: Verbosity::Silent,
            force_frame_pointer: false,
            enable_licm: true,
            enable_dce: true,
            enable_store_forwarding: true,
            enable_dse: true,
            enable_phi_simplify: false,
            enable_inlining: true,
            max_inline_depth: 3,
            max_inline_nodes: 50,
            inline_cost_threshold: 100,
        }
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions::o1()
    }
}

#[derive(Debug)]
pub struct CompileError {
    pub phase: String,
    pub message: String,
    pub location: Option<IrLocation>,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "phase '{}': {}", self.phase, self.message)?;
        if let Some(loc) = &self.location {
            write!(f, " (in function '{}'", loc.function)?;
            if let Some(b) = loc.block {
                write!(f, ", block {b}")?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl std::error::Error for CompileError {}

#[derive(Debug)]
pub struct IrLocation {
    pub function: String,
    pub block: Option<u32>,
    pub inst: Option<usize>,
}

pub trait DiagnosticSink {
    fn phase_stats(&mut self, phase: &str, stats: &str);
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Returns true if any block in `func` contains a Call effectful operation.
fn func_has_calls(func: &Function) -> bool {
    func.blocks.iter().any(|b| {
        b.ops
            .iter()
            .any(|op| matches!(op, EffectfulOp::Call { .. }))
    })
}

// ── Shared egraph + extraction phases ────────────────────────────────────────

use crate::egraph::egraph::EGraph;
use crate::egraph::extract::ExtractionResult;

/// Run the e-graph rewrite rules and cost-based extraction (phases 1-2).
///
/// Shared between `compile()` and `compile_to_ir_string()`.
pub(super) fn run_egraph_and_extract(
    func: &Function,
    egraph: &mut EGraph,
    opts: &CompileOptions,
) -> Result<(BTreeMap<(BlockId, u32), ClassId>, ExtractionResult), CompileError> {
    let egraph_opts = EGraphOptions {
        iteration_limit: opts.saturation_limit,
        max_classes: 500_000,
    };
    crate::egraph::algebraic::propagate_block_params(func, egraph);
    run_phases(egraph, &egraph_opts).map_err(|e| CompileError {
        phase: "egraph".into(),
        message: e,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;
    crate::egraph::algebraic::propagate_block_params(func, egraph);
    crate::egraph::algebraic::apply_algebraic_rules(egraph);
    egraph.rebuild();

    // Close instruction selection over everything the rules above produced.
    // Extraction has no fallback for a class without a machine op, so this is
    // a correctness step, not part of the optimization budget.
    crate::egraph::phases::saturate_isel(egraph).map_err(|e| CompileError {
        phase: "isel".into(),
        message: e,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

    let block_param_map = egraph.block_param_classes();

    let mut all_roots = collect_roots(func, egraph);
    all_roots.extend(block_param_map.values().copied());
    all_roots.sort_by_key(|c| c.0);
    all_roots.dedup();

    let cost_model = CostModel::new(opts.opt_goal);
    let extraction = extract(egraph, &all_roots, &cost_model).map_err(|e| CompileError {
        phase: "extraction".into(),
        message: e.to_string(),
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

    Ok((block_param_map, extraction))
}

/// What the IR-level passes hand to the machine-level half of the pipeline.
pub(super) struct IrPasses {
    pub extra_roots: licm::ExtraRoots,
    pub block_param_map: BTreeMap<(BlockId, u32), ClassId>,
    pub extraction: ExtractionResult,
}

/// Run every IR-level pass: forwarding, DSE, LICM, saturation + extraction, DCE2.
///
/// Shared by `compile()` and `compile_to_ir_string()` so an `--emit-ir` dump
/// always reflects the IR a real compile would produce.
///
/// Each merging pass is followed by [`canonicalize_class_refs`], and each pass
/// boundary by a `BLITZ_VERIFY` check naming that stage.
pub(super) fn run_ir_passes(
    func: &mut Function,
    egraph: &mut EGraph,
    opts: &CompileOptions,
    sink: &mut Option<&mut dyn DiagnosticSink>,
) -> Result<IrPasses, CompileError> {
    crate::verify::verify_stage("ir-entry", func, egraph);

    // Redundant block parameters, first: every later pass keyed on
    // `(BlockId, param_index)` needs the arity final, and every later pass gets
    // simpler input for it.
    if opts.enable_phi_simplify {
        let removed = phi_simplify::simplify_block_params(func, egraph);
        if removed > 0 {
            canonicalize_class_refs(func, egraph);
        }
        if crate::trace::is_enabled("egraph") && crate::trace::fn_matches(&func.name) {
            tracing::debug!(
                target: "blitz::egraph",
                "[{}] phi-simplify: removed {removed} block parameter(s)", func.name,
            );
        }
        crate::verify::verify_stage("phi-simplify", func, egraph);
    }

    // Store-to-load / load-to-load forwarding: intra-block, pre-LICM so
    // hoisting and saturation both benefit from fewer memory ops.
    if opts.enable_store_forwarding {
        let alias = AliasInfo::new();
        forward::run_forwarding(func, egraph, &alias);
        canonicalize_class_refs(func, egraph);
        crate::verify::verify_stage("forwarding", func, egraph);
    }

    // Dead store elimination: runs after forwarding (more forwarded loads =
    // fewer pending-store "observed by load" cancellations = more kills).
    if opts.enable_dse {
        let alias = AliasInfo::new();
        dse::run_dse(func, egraph, &alias);
        crate::verify::verify_stage("dse", func, egraph);
    }

    // LICM: detect loops, insert preheaders, identify invariant classes.
    let extra_roots = if opts.enable_licm {
        let roots = licm::run_licm(func, egraph);
        canonicalize_class_refs(func, egraph);
        crate::verify::verify_stage("licm", func, egraph);
        roots
    } else {
        Default::default()
    };

    // Phases 1-2: E-graph rewrites and cost-based extraction.
    // Extraction only reads `func`; DCE2 below needs it mutable again.
    let (block_param_map, extraction) = run_egraph_and_extract(func, egraph, opts)?;
    canonicalize_class_refs(func, egraph);
    crate::verify::verify_stage("saturation+extraction", func, egraph);

    if let Some(s) = sink.as_mut() {
        s.phase_stats(
            "egraph",
            &format!(
                "classes={}, nodes={}",
                egraph.class_count(),
                egraph.node_count()
            ),
        );
        s.phase_stats(
            "extraction",
            &format!("classes_extracted={}", extraction.choices.len()),
        );
    }

    // DCE2: constant branch folding, unreachable block elimination, dead loads.
    // Must run before the caller freezes `func` and builds index-keyed structures.
    let extra_roots = if opts.enable_dce {
        let roots = dce::run_dce2_with_extra_roots(func, egraph, &extraction, extra_roots);
        crate::verify::verify_stage("dce2", func, egraph);
        roots
    } else {
        extra_roots
    };

    Ok(IrPasses {
        extra_roots,
        block_param_map,
        extraction,
    })
}

// ── compile() ────────────────────────────────────────────────────────────────

/// Compile a single function to an object file.
///
/// Consumes the `Function` (including its embedded e-graph).
pub fn compile(
    mut func: Function,
    opts: &CompileOptions,
    mut sink: Option<&mut dyn DiagnosticSink>,
) -> Result<ObjectFile, CompileError> {
    crate::trace::init_tracing();

    // Compute user stack space in 8-byte units. Each slot may be larger than
    // 8 bytes (e.g. string literal buffers), so sum actual sizes rounded up.
    let user_stack_slots: u32 = func.stack_slots.iter().map(|s| s.size.div_ceil(8)).sum();
    let mut egraph = func
        .egraph
        .take()
        .expect("Function must contain an EGraph; use FunctionBuilder::finalize()");

    let IrPasses {
        extra_roots,
        block_param_map,
        extraction,
    } = run_ir_passes(&mut func, &mut egraph, opts, &mut sink)?;

    // `func` stays mutable until linearization has chosen a VReg for every block
    // argument and `commit_terminator_arg_vregs` has written that choice into the
    // terminators; from there the CFG is read-only.

    // Must be after DCE2, which changes what the indices are.
    let block_id_to_idx = cfg::block_id_to_idx(&func);

    // Detect whether this function contains any call instructions (must be after DCE2).
    let has_calls = func_has_calls(&func);

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
    let rpo_order = compute_rpo(&func);
    // Predecessor counts per block index. Used by block_param_fixup to
    // distinguish loop headers / merge points (multi-pred, need phi storage)
    // from pass-through blocks (single-pred, the block param IS its sole
    // predecessor's argument and doesn't need a fresh VReg).
    let block_preds = cfg::predecessor_indices(&func);

    // Map (BlockId, param_idx) -> fresh VReg for block params whose canonical
    // VReg was emitted by a prior block. This prevents the e-graph from merging
    // outer and inner loop header params into the same register.
    let mut block_param_vreg_overrides: BTreeMap<(BlockId, u32), VReg> = BTreeMap::new();

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
    //
    // Deliberately NOT folded into `block_param_vreg_overrides`: that map also
    // feeds an `insert_single` into each block's snapshot, which would collapse
    // a class's segments to one full-range entry and discard what the splitter
    // recorded.
    let mut block_param_vregs: BTreeMap<(BlockId, u32), VReg> = BTreeMap::new();

    let idom = compute_idom(&func, &rpo_order);
    let mut class_emitted_in: BTreeMap<ClassId, usize> = BTreeMap::new();

    // Build per-block VRegInst lists in RPO order, stored by block index.
    let mut block_vreg_insts: Vec<Vec<VRegInst>> = vec![Vec::new(); func.blocks.len()];
    // Snapshot of class_to_vreg AT THE END of each block's processing (before
    // `removed` is restored). Captures the block-local view: classes re-emitted
    // in a block point to that block's VReg, not the globally-restored one.
    let mut block_class_to_vreg_snapshot: Vec<ClassVRegMap> =
        vec![ClassVRegMap::new(); func.blocks.len()];
    for &block_idx in &rpo_order {
        // Remove classes emitted in non-dominating blocks so they get fresh VRegs.
        // Also remove flags-typed classes from ALL prior blocks: EFLAGS cannot
        // survive cross-block boundaries because any arithmetic instruction
        // clobbers them.
        let removable_classes: Vec<ClassId> = class_emitted_in
            .iter()
            .filter(|(cid, emitter)| {
                if !dominates(**emitter, block_idx, &idom) {
                    return true;
                }
                // Flags-typed classes must be re-emitted per-block.
                if **emitter != block_idx {
                    let ty = &egraph.classes[cid.0 as usize].ty;
                    if matches!(ty, Type::Flags)
                        || matches!(ty, Type::Pair(_, b) if **b == Type::Flags)
                    {
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
        let roots = collect_block_roots(block, &egraph);
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
            vreg_insts_for_block(&extraction, &all_roots, &mut class_to_vreg, &mut next_vreg);

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
                        inst.op = Op::BlockParam(
                            block_id,
                            pidx,
                            block.param_types[pidx as usize].clone(),
                        );
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
                            op: Op::BlockParam(
                                block_id,
                                pidx,
                                block.param_types[pidx as usize].clone(),
                            ),
                            operands: vec![],
                        });
                        block_param_vreg_overrides.insert((block_id, pidx), fresh_vreg);
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
    let mut vreg_types = build_vreg_types(&class_to_vreg, &egraph);
    for snapshot in &block_class_to_vreg_snapshot {
        vreg_types.extend(build_vreg_types(snapshot, &egraph));
    }

    // Insert types for fresh block param VRegs allocated above.
    for (&(bid, pidx), &fresh_vreg) in &block_param_vreg_overrides {
        let block = &func.blocks[block_id_to_idx[&bid]];
        let ty = block.param_types[pidx as usize].clone();
        vreg_types.insert(fresh_vreg, ty);
    }

    // Commit linearization's choice into the CFG: from here a block argument is
    // a VReg the terminator names, not a class every consumer resolves for
    // itself. Before scheduling, so the schedules are built from what this
    // wrote, and before the splitter, whose operand rewriting is what makes a
    // pressure decision stick.
    let mut rpo_pos = vec![0usize; func.blocks.len()];
    for (pos, &idx) in rpo_order.iter().enumerate() {
        rpo_pos[idx] = pos;
    }
    commit_terminator_arg_vregs(
        &mut func,
        &egraph,
        &class_to_vreg,
        &block_class_to_vreg_snapshot,
        &block_param_map,
        &block_param_vreg_overrides,
        &rpo_pos,
    )
    .map_err(|message| CompileError {
        phase: "terminator-args".into(),
        message,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

    // NOW freeze func for the rest of the pipeline.
    let func = &func;

    // Phase 4: Schedule per block (indexed by block index, same as block_vreg_insts).
    let mut block_schedules: Vec<Vec<ScheduledInst>> = vec![Vec::new(); func.blocks.len()];
    let mut total_insts = 0usize;
    for (block_idx, insts) in block_vreg_insts.iter().enumerate() {
        let dag = ScheduleDag::build(insts);
        let sched = schedule(&dag);
        total_insts += sched.len();
        block_schedules[block_idx] = sched;
    }

    if let Some(s) = sink.as_mut() {
        s.phase_stats("schedule", &format!("insts={total_insts}"));
    }

    // Phase 4b: Reorder each block's schedule to respect effectful op barriers.
    //
    // Effectful ops (loads, stores, calls) impose ordering constraints on pure ops:
    // pure ops that consume a LoadResult must come after the corresponding Load,
    // and pure ops that are inputs to a Call must come before the Call. The
    // scheduler doesn't know about effectful ops, so we reorder the schedule
    // here so the regalloc sees correct liveness.
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let has_branch = block
            .ops
            .last()
            .is_some_and(|op| matches!(op, EffectfulOp::Branch { .. }));

        if block.non_term_count() == 0 && !has_branch {
            continue; // No effectful ops and no branch to constrain ordering.
        }

        let (vreg_to_result_of_barrier, vreg_to_arg_of_barrier) = if block.non_term_count() > 0 {
            build_barrier_context(block, block_idx, &egraph, &class_to_vreg)
        } else {
            (BTreeMap::new(), BTreeMap::new())
        };

        let sched = &block_schedules[block_idx];
        let vreg_group =
            assign_barrier_groups(sched, &vreg_to_result_of_barrier, &vreg_to_arg_of_barrier);

        // Identify the branch condition's flags-producing instruction so it
        // sorts to the end of its barrier group. On x86, any ALU instruction
        // clobbers EFLAGS, so the flags-producing instruction must be the
        // last ALU op before the terminator. We only move the immediate
        // flags chain (proj1 + its parent ALU op), not the full transitive
        // operand tree, to avoid disrupting scheduling of shared operands.
        let mut branch_cond_chain: BTreeSet<VReg> = BTreeSet::new();
        if let Some(EffectfulOp::Branch { cond, .. }) = block.ops.last() {
            let canon = egraph.unionfind.find_immutable(*cond);
            if let Some(vreg) = class_to_vreg.lookup(canon, ProgramPoint::block_exit(block_idx)) {
                // Add the flags VReg (proj1).
                branch_cond_chain.insert(vreg);
                // Find the instruction that produces it and add its parent
                // (the ALU op that sets EFLAGS, e.g. x86_sub).
                for inst in sched {
                    if inst.dst == vreg {
                        if matches!(inst.op, Op::Proj1) {
                            for &op in &inst.operands {
                                branch_cond_chain.insert(op);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Stable-sort by group to reorder while preserving within-group order.
        // Barrier results (LoadResult/CallResult) sort to the FRONT of their
        // group: their values are produced by effectful ops at the group
        // boundary, so the register is occupied from the start of the group.
        // Placing them after pure ops would let the regalloc think the register
        // is free, causing incorrect reuse and clobbering.
        // Branch condition chain sorts to the END of its group to prevent
        // other ALU ops from clobbering EFLAGS between the flags-producing
        // instruction and the branch terminator.
        let mut indexed: Vec<(usize, &ScheduledInst)> = sched.iter().enumerate().collect();
        indexed.sort_by_key(|(orig_idx, inst)| {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            let param_order: u8 = match inst.op {
                Op::Param(_, _) => 0,
                Op::LoadResult(_, _) | Op::CallResult(_, _) => 1,
                // Spill reloads must happen early in their consumer group,
                // BEFORE any op that uses the reloaded value. Pushing the
                // SpillLoad's orig_idx to the end of the block (via barrier.rs)
                // would otherwise place it after its consumer under the
                // default param_order=2 tier. param_order=1 places it right
                // after the group's barrier result and before pure ops.
                Op::SpillLoad(_) | Op::XmmSpillLoad(_) => 1,
                _ if branch_cond_chain.contains(&inst.dst) => 3,
                _ => 2,
            };
            (g, param_order, *orig_idx)
        });
        let reordered: Vec<ScheduledInst> =
            indexed.into_iter().map(|(_, inst)| inst.clone()).collect();
        block_schedules[block_idx] = reordered;

        if crate::trace::is_enabled("sched") && crate::trace::fn_matches(&func.name) {
            tracing::debug!(
                target: "blitz::sched",
                "[{}] block {block_idx} after barrier sort:\n{}",
                func.name,
                crate::trace::format_schedule(&block_schedules[block_idx], Some(&vreg_group)),
            );
        }
    }

    // Hoist Param and BlockParam ops to the front of their block's schedule.
    //
    // Neither computes anything: the value is already in its register when the
    // block starts, put there by the ABI or by a predecessor's phi copy. Their
    // pseudo-ops are still where scheduling left them, though, and liveness reads
    // a def position as the start of a live range -- so a block param whose
    // pseudo-op the backward pass pulled down next to its use looked dead over
    // the earlier part of its own block, and the allocator handed its register to
    // something defined there:
    //
    //   movsd xmm3,xmm4     ; x + y, into the register holding param 4
    //   ...
    //   addsd xmm1,xmm3     ; wanted param 4, reads x + y
    //
    // Hoisting makes the def position match where the value actually arrives.
    for sched in block_schedules.iter_mut() {
        sched.sort_by_key(|inst| {
            if matches!(inst.op, Op::Param(_, _) | Op::BlockParam(_, _, _)) {
                0u8
            } else {
                1
            }
        });
    }

    // Phase 5: Register allocation -- per-block with cross-block live range splitting.
    //
    // Single-block fast path: skip global liveness and run allocate() directly.
    // Multi-block path: compute global liveness, assign cross-block spill slots,
    // rewrite each block to insert spill/reload code at boundaries, then run
    // allocate() per block and merge results.
    let entry_has_calls = func.blocks[0]
        .ops
        .iter()
        .any(|op| matches!(op, EffectfulOp::Call { .. }));
    let func_arg_locs = crate::x86::abi::assign_args(&func.param_types);
    let param_vregs = assign_param_vregs_from_map(
        func,
        &class_to_vreg,
        &egraph,
        entry_has_calls,
        &func_arg_locs,
    );

    // Compute loop depths from the CFG for spill selection.
    let loop_depths = compute_loop_depths(func, &block_schedules);

    // Block params the splitter routed through a stack slot.
    // Passed to lower_terminator so predecessor terminators emit slot stores.
    let mut slot_spilled_params: crate::compile::split::BlockParamSlotMap =
        std::collections::BTreeMap::new();
    // End-of-block SpillLoad VRegs inserted by cross-block spills. The lowering
    // pass forces these into the trailing barrier group so they execute after all
    // calls in their block (preventing loads from uninitialized stack slots).

    // The final `phi_uses`, kept for `verify::verify_register_sharing` after
    // allocation: it is the terminator half of liveness, and the check needs the
    // same one the allocator was given.
    let mut verify_phi_uses: Vec<BTreeSet<VReg>> = Vec::new();
    // The block parameters, for the same reason: a parameter is written by the
    // phi copy at the edge, so it is not live at a predecessor's exit unless the
    // predecessor's terminator passes it, which `verify_phi_uses` records.
    let mut verify_block_params: Vec<BTreeSet<VReg>> = Vec::new();
    // And the copy pairs, so the check exempts the same phi-related VRegs
    // coalescing was allowed to merge.
    let mut verify_copy_pairs: Vec<(VReg, VReg)> = Vec::new();

    // Every pass that spills draws its slot numbers from here, so no two name the
    // same 8-byte cell of the frame and each slot can say which pass owns it.
    let mut slots = SlotAllocator::new();

    // Single-block fast path skips global liveness.
    let (regalloc_result, block_rewritten, coalesce_aliases) = if func.blocks.len() == 1 {
        // --- Single-block fast path ---
        let mut all_scheduled: Vec<ScheduledInst> =
            block_schedules.iter().flatten().cloned().collect();

        // Populate effectful-op operands right before regalloc so it sees
        // effectful op operand liveness at the correct barrier positions.
        {
            let block = &func.blocks[0];
            let non_term_count = block.non_term_count();
            if non_term_count > 0 {
                let non_term_ops = &block.ops[..non_term_count];
                let (result_map, arg_map) =
                    build_barrier_context(block, 0, &egraph, &class_to_vreg);
                let mut vreg_group = assign_barrier_groups(&all_scheduled, &result_map, &arg_map);
                populate_effectful_operands(
                    &mut all_scheduled,
                    non_term_ops,
                    0,
                    &egraph,
                    &class_to_vreg,
                    &mut vreg_group,
                    &mut next_vreg,
                );

                if crate::trace::is_enabled("sched") && crate::trace::fn_matches(&func.name) {
                    tracing::debug!(
                        target: "blitz::sched",
                        "[{}] single-block after markers:\n{}",
                        func.name,
                        crate::trace::format_schedule(&all_scheduled, Some(&vreg_group)),
                    );
                }
            }
        }

        let mut live_out: BTreeSet<VReg> = BTreeSet::new();
        collect_phi_source_vregs(func, &mut live_out);
        // Add Ret operands to live_out. Ret is the terminator (no barrier
        // instruction) so its operands must survive until end of block.
        if let Some(EffectfulOp::Ret { val: Some(cid) }) = func.blocks[0].ops.last() {
            let canon = egraph.unionfind.find_immutable(*cid);
            if let Some(vreg) = class_to_vreg.lookup(canon, ProgramPoint::block_exit(0)) {
                live_out.insert(vreg);
            }
        }
        for &(vreg, _reg) in &param_vregs {
            live_out.insert(vreg);
        }

        let mut all_param_vregs = param_vregs.clone();
        add_shift_precolors(&all_scheduled, &mut all_param_vregs);
        add_div_precolors(&all_scheduled, &mut all_param_vregs);
        add_call_precolors_for_block(
            &func.blocks[0],
            0,
            &egraph,
            &class_to_vreg,
            &mut all_param_vregs,
            &mut live_out,
        );
        // Coalescing's copy pairs. One block reaches no other, so the only edge
        // this can find is a self-loop.
        let copy_pairs = compute_copy_pairs(
            func,
            &class_to_vreg,
            &egraph,
            &block_param_map,
            &block_param_vreg_overrides,
        );
        let result = allocate(
            &all_scheduled,
            &all_param_vregs,
            &live_out,
            &copy_pairs,
            &loop_depths,
            &mut slots,
            opts.force_frame_pointer,
            &func.name,
        )
        .map_err(|e| CompileError {
            phase: "regalloc".into(),
            message: e,
            location: Some(IrLocation {
                function: func.name.clone(),
                block: None,
                inst: None,
            }),
        })?;

        for inst in &result.insts {
            for &op in &inst.operands {
                debug_assert!(
                    result.vreg_to_reg.contains_key(&op),
                    "operand VReg {:?} has no register assignment",
                    op
                );
            }
        }
        let rewritten = vec![result.insts.clone()];
        let aliases: BTreeMap<VReg, VReg> = BTreeMap::new();
        (result, rewritten, aliases)
    } else {
        // --- Multi-block path: the function-scope register allocator ---

        // Step 1: Compute CFG successors. Terminator uses come from the
        // `Op::TerminatorArgs` operands once those exist, below.
        let cfg_succs = cfg::successor_indices(func);

        // ORDER: before `populate_effectful_operands`, which appends the trailing
        // liveness operands and dedupes them. `add_call_precolors_for_block` reads
        // `EffectfulOp::Call`'s args positionally to pin each to its ABI register,
        // so it needs the arg list as the CFG states it.
        let mut call_arg_precolors: Vec<(VReg, Reg)> = Vec::new();
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let mut dummy_live_out: std::collections::BTreeSet<VReg> =
                std::collections::BTreeSet::new();
            add_call_precolors_for_block(
                block,
                block_idx,
                &egraph,
                &class_to_vreg,
                &mut call_arg_precolors,
                &mut dummy_live_out,
            );
        }

        // Shorten a barrier result's live range where its consumer is two or more
        // barrier groups away, by spilling it at the def and reloading it at the
        // use. First of the three passes that spill in this function, all drawing
        // their slots from `slots`.
        {
            let mut early_next_vreg = next_vreg;
            for (block_idx, block) in func.blocks.iter().enumerate() {
                let non_term_count = block.non_term_count();
                if non_term_count > 0 {
                    let (result_map, arg_map) =
                        build_barrier_context(block, block_idx, &egraph, &class_to_vreg);
                    let mut vreg_group =
                        assign_barrier_groups(&block_schedules[block_idx], &result_map, &arg_map);
                    insert_early_barrier_spills(
                        &mut block_schedules[block_idx],
                        &result_map,
                        &arg_map,
                        &mut vreg_group,
                        &vreg_types,
                        &mut early_next_vreg,
                        &mut slots,
                    );
                }
            }
            next_vreg = early_next_vreg;
        }

        // Populate effectful-op operands onto barrier instructions (LoadResult,
        // CallResult, StoreBarrier, VoidCallBarrier) in each block's schedule
        // BEFORE global liveness, so compute_global_liveness sees them as regular
        // instruction operands and includes them in cross-block liveness.
        // This MUST happen AFTER call_arg_precolors collection.
        //
        // Resolved through each block's own snapshot, not the global map. A class
        // re-emitted per block has one VReg per block and the global map holds
        // whichever was restored last, so an `if`/`else` pair that both call
        // `printf("%d\n", ...)` would record the other arm's VReg for the format
        // string -- and since Phase 7 now trusts these operands as the record of
        // what each op reads, that names a register holding something else.
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let non_term_count = block.non_term_count();
            if non_term_count > 0 {
                let non_term_ops = &block.ops[..non_term_count];
                let block_map = &block_class_to_vreg_snapshot[block_idx];
                let (result_map, arg_map) =
                    build_barrier_context(block, block_idx, &egraph, block_map);
                let mut vreg_group =
                    assign_barrier_groups(&block_schedules[block_idx], &result_map, &arg_map);
                populate_effectful_operands(
                    &mut block_schedules[block_idx],
                    non_term_ops,
                    block_idx,
                    &egraph,
                    block_map,
                    &mut vreg_group,
                    &mut next_vreg,
                );
            }

            // The terminator's arguments, as operands, for the same reason the
            // barrier ops carry theirs: so the splitter and coalescing rewrite
            // them instead of three passes re-deriving them from a map those
            // passes mutate. A copy of what the CFG names, not a second
            // resolution of it.
            if let Some(term) = block.ops.last() {
                // A `Ret`'s value is the one argument still named by class, so
                // this is the one still resolved. A parameter of this block that
                // linearization gave a fresh VReg is not in the snapshot, which
                // predates it, so the overrides answer first.
                let param_override_vregs: BTreeMap<ClassId, VReg> = block_param_vreg_overrides
                    .iter()
                    .filter(|((bid, _), _)| *bid == block.id)
                    .filter_map(|(&(bid, pidx), &fresh)| {
                        let cid = block_param_map.get(&(bid, pidx))?;
                        Some((egraph.unionfind.find_immutable(*cid), fresh))
                    })
                    .collect();
                crate::compile::barrier::append_terminator_args(
                    &mut block_schedules[block_idx],
                    term,
                    block_idx,
                    &egraph,
                    &block_class_to_vreg_snapshot[block_idx],
                    &param_override_vregs,
                    &mut next_vreg,
                )
                .map_err(|message| CompileError {
                    phase: "terminator-args".into(),
                    message,
                    location: Some(IrLocation {
                        function: func.name.clone(),
                        block: Some(block.id),
                        inst: None,
                    }),
                })?;
            }
        }

        // Every effectful-op role operand now exists in both representations and
        // must name the same VReg. Checked here, before the splitter, so a
        // disagreement is attributable to construction.
        crate::verify::verify_cfg_schedule_agreement_stage(
            "effectful-operands",
            func,
            &egraph,
            &block_schedules,
            &block_class_to_vreg_snapshot,
        );

        // Terminator uses, read straight off the schedules. `Op::TerminatorArgs`
        // is the record of what each terminator consumes: the splitter rewrites
        // its operands and coalescing renames them, so recomputing this after
        // either pass gives that pass's answer rather than a second, independent
        // derivation that can disagree with it.
        let terminator_uses = |schedules: &[Vec<ScheduledInst>]| -> Vec<BTreeSet<VReg>> {
            schedules
                .iter()
                .map(|s| {
                    barrier::terminator_arg_operands(s)
                        .into_iter()
                        .map(|(_, v)| v)
                        .collect()
                })
                .collect()
        };
        let mut phi_uses = terminator_uses(&block_schedules);

        // Pressure-driven splitter.
        // CRITICAL ORDER: apply_plan_to must run BEFORE collect_block_param_vregs_per_block.
        // The splitter may truncate segments, which affects what block params are found.
        // One round of splitting only lowers pressure at the points it looked
        // at, and the reloads it inserts are themselves live somewhere. Re-plan
        // against the rewritten schedules until no overshoot is left or a round
        // finds nothing to do -- a single pass leaves the allocator facing an
        // overshoot it can only report as a failure.
        for _round in 0..split::MAX_SPLIT_ROUNDS {
            use crate::regalloc::coloring::{AVAILABLE_XMM_COLORS, available_gpr_colors};

            let gpr_budget = available_gpr_colors(opts.force_frame_pointer);
            let xmm_budget = AVAILABLE_XMM_COLORS;

            // Compute global liveness to seed the splitter's per-block backward scans.
            let split_global_liveness = crate::regalloc::global_liveness::compute_global_liveness(
                &block_schedules,
                &cfg_succs,
                &phi_uses,
            );

            let split_cost_model = CostModel::new(opts.opt_goal);
            let plan = split::plan_splits(
                &block_schedules,
                &class_to_vreg,
                &extraction,
                &egraph,
                &split_cost_model,
                &split_global_liveness,
                gpr_budget,
                xmm_budget,
                next_vreg,
                &mut slots,
                &loop_depths,
                func,
                &block_param_map,
                &slot_spilled_params,
            );
            // An empty plan still goes through `apply_plan_to`: that is what
            // bumps `split_generation`, and `collect_block_param_vregs_per_block`
            // asserts the splitter has committed at least once.
            let converged = plan.is_empty();
            // Accumulate the maps across rounds. The plan keeps its
            // `slot_spilled_params`: `apply_plan_to` reads them to truncate each
            // param's segment past block entry, which is what stops a register
            // being allocated to a value that lives in a slot.
            slot_spilled_params.extend(plan.slot_spilled_params.clone());

            // Build old→new VReg remap restricted to CALL-ARG positions so we
            // can update call_arg_precolors after apply_plan_to. The precolors
            // were collected before the splitter ran; if the splitter rewrites a
            // call-arg VReg to a reload VReg, the precolor must follow the reload.
            //
            // Restriction to call-arg positions avoids using the wrong reload VReg
            // when the same old VReg is rewritten in multiple blocks (non-call uses
            // create reload VRegs that should NOT inherit the ABI precolor).
            let call_arg_vreg_set: std::collections::BTreeSet<VReg> =
                call_arg_precolors.iter().map(|(v, _)| *v).collect();
            let mut vreg_remap: BTreeMap<VReg, VReg> = BTreeMap::new();
            for &(bi, ii, oi, new_vreg) in &plan.operand_rewrites {
                // Only update if the old VReg is a call-arg precolor candidate.
                if let Some(old_vreg) = block_schedules
                    .get(bi)
                    .and_then(|s| s.get(ii))
                    .and_then(|inst| inst.operands.get(oi))
                    .copied()
                    && call_arg_vreg_set.contains(&old_vreg)
                {
                    // Only keep first entry (the call-site reload VReg; later
                    // entries for the same old VReg are non-call-site reloads).
                    vreg_remap.entry(old_vreg).or_insert(new_vreg);
                }
            }

            let applied = split::apply_plan_to(
                &mut block_schedules,
                &mut class_to_vreg,
                &mut next_vreg,
                plan,
            );

            // The per-block snapshots were taken during linearization, before
            // the splitter ran, so they still map every class to the VReg it
            // had before any spill. Phase 7 resolves effectful-op operands (a
            // Load's address, a Store's value, a call argument) through them,
            // and those operands are ClassIds in the CFG that no operand
            // rewrite reaches -- so without the splitter's segments a spilled
            // address resolves to the register it occupied *before* the spill,
            // and the load reads a register the reload never wrote.
            for snapshot in block_class_to_vreg_snapshot.iter_mut() {
                applied.replay_onto(snapshot);
            }

            if crate::trace::is_enabled("split") && crate::trace::fn_matches(&func.name) {
                for (block_idx, sched) in block_schedules.iter().enumerate() {
                    tracing::debug!(
                        target: "blitz::split",
                        "[{}] block {block_idx} after split plan:\n{}",
                        func.name,
                        crate::trace::format_schedule(sched, None),
                    );
                }
            }

            // Update call_arg_precolors: transfer each precolor to its reload VReg.
            if !vreg_remap.is_empty() {
                for (precolor_vreg, _reg) in call_arg_precolors.iter_mut() {
                    if let Some(&new_vreg) = vreg_remap.get(precolor_vreg) {
                        *precolor_vreg = new_vreg;
                    }
                }
            }

            // An argument whose destination parameter lives in a slot leaves the
            // terminator's operand list, and becomes its own store unless the slot
            // already holds its value.
            //
            // Leaving it as an operand defeats the routing: the operand list is
            // the parallel copy, which needs every argument readable at one point,
            // so sixteen slot-bound arguments still ask for sixteen registers at
            // once and the clique comes back one instruction later. A store to a
            // slot clobbers no other argument's register, so these go one at a
            // time, and as ordinary instructions with one operand each that is
            // exactly what liveness then sees.
            //
            // Decided per argument, by the destination it feeds -- never by
            // "this VReg belongs to some routed parameter". An edge can pass one
            // routed parameter's VReg to a different parameter, and dropping that
            // operand for the wrong reason left the destination's slot unwritten
            // while every use reloaded from it.
            //
            // `build_phi_copies` needs no matching change: an argument index with
            // no operand is already the signal that the value travels through a
            // slot and the store is already in this block.
            for (block_idx, block) in func.blocks.iter().enumerate() {
                let terminator = block.ops.last().expect("block must have terminator");
                let dests = barrier::terminator_arg_destinations(terminator);
                // `(arg VReg, destination info)`, and whether a store is needed:
                // an argument that IS its destination parameter has nothing to
                // store, the slot holds that value already.
                let routed: Vec<(u32, VReg, (BlockId, u32), split::SlotSpilledParamInfo)> =
                    barrier::terminator_arg_operands(&block_schedules[block_idx])
                        .into_iter()
                        .filter_map(|(arg_idx, vreg)| {
                            let &(target, pidx) = dests.get(arg_idx as usize)?;
                            let info = slot_spilled_params.get(&(target, pidx))?.clone();
                            Some((arg_idx, vreg, (target, pidx), info))
                        })
                        .collect();
                if routed.is_empty() {
                    continue;
                }
                let stores: Vec<(VReg, split::SlotSpilledParamInfo)> = routed
                    .iter()
                    .filter(|(_, vreg, dest, info)| {
                        *vreg != info.vreg
                            && block_param_vreg_overrides
                                .get(dest)
                                .is_none_or(|&ov| *vreg != ov)
                    })
                    .map(|(_, vreg, _, info)| (*vreg, info.clone()))
                    .collect();
                let schedule = &mut block_schedules[block_idx];
                let drop_args: BTreeSet<u32> = routed.iter().map(|(idx, _, _, _)| *idx).collect();
                barrier::remove_terminator_arg_operands(schedule, &drop_args);
                if stores.is_empty() {
                    continue;
                }

                // Each store goes immediately after the last point its value is
                // needed for anything else -- its definition, or a later use.
                // Putting them all at the end instead relieves nothing: the value
                // is live until the instruction that reads it, so sixteen stores
                // at the end keep sixteen registers occupied until the end,
                // exactly as the terminator operands did.
                let last_needed = |vreg: VReg, sched: &[ScheduledInst]| -> usize {
                    sched
                        .iter()
                        .enumerate()
                        .filter(|(_, inst)| {
                            !matches!(inst.op, Op::TerminatorArgs(_))
                                && (inst.dst == vreg || inst.operands.contains(&vreg))
                        })
                        .map(|(i, _)| i + 1)
                        .max()
                        .unwrap_or(0)
                };
                let mut planned: Vec<(usize, ScheduledInst)> = stores
                    .into_iter()
                    .map(|(vreg, info)| {
                        let store = ScheduledInst {
                            op: match info.reg_class {
                                crate::x86::reg::RegClass::XMM => Op::XmmSpillStore(info.slot),
                                crate::x86::reg::RegClass::GPR => Op::SpillStore(info.slot),
                            },
                            dst: VReg(next_vreg),
                            operands: vec![vreg],
                        };
                        next_vreg += 1;
                        (last_needed(vreg, schedule), store)
                    })
                    .collect();
                // Descending, so an insertion never moves a position not yet used.
                planned.sort_by(|a, b| b.0.cmp(&a.0));
                for (at, store) in planned {
                    schedule.insert(at.min(schedule.len()), store);
                }
            }

            // The plan rewrote terminator operands to its reload VRegs, and the
            // slot-routed ones are gone, so re-reading the schedules is what
            // makes a spilled value stop being live out.
            phi_uses = terminator_uses(&block_schedules);

            if converged {
                break;
            }
        }

        // And again on the splitter's output, which is the state Phase 7 asks
        // about. Reported rather than enforced: the splitter's segments do not
        // reliably cover the point its own reload is consumed at, so the map is
        // already the weaker of the two answers here. See
        // `verify::report_cfg_schedule_agreement`.
        crate::verify::report_cfg_schedule_agreement(
            "split",
            func,
            &egraph,
            &block_schedules,
            &block_class_to_vreg_snapshot,
        );

        // Step 3: Determine block params per block (passed to allocate_global).
        // CRITICAL ORDER: must run AFTER apply_plan_to (splitter output committed).
        let mut block_param_vregs_per_block =
            crate::regalloc::global_liveness::collect_block_param_vregs_per_block(
                func,
                &egraph,
                &block_param_map,
                &class_to_vreg,
            );

        // Include fresh block param VRegs in the per-block sets.
        for (&(bid, _pidx), &fresh_vreg) in &block_param_vreg_overrides {
            let block_idx = block_id_to_idx[&bid];
            block_param_vregs_per_block[block_idx].insert(fresh_vreg);
        }

        // And what linearization recorded, for every param the class map cannot
        // name. `collect_block_param_vregs_per_block` finds a param only where a
        // segment still covers the block entry, but `lower_terminator` falls back
        // to `block_param_vregs`, so a param the splitter truncated is a param
        // there and not one here. The allocator then never learns it is written
        // at block entry: it draws no interference edge to its siblings, and
        // coalescing is free to merge two parameters of one block into a single
        // register. Both phi copies then target that register and the second
        // overwrites the first.
        for (&(bid, pidx), &vreg) in &block_param_vregs {
            if slot_spilled_params.contains_key(&(bid, pidx)) {
                continue;
            }
            block_param_vregs_per_block[block_id_to_idx[&bid]].insert(vreg);
        }

        // Remove slot-spilled params from block_param_vregs_per_block.
        // Slot-spilled params have no register: they are written via slot stores
        // by predecessor terminators and loaded on use. Adding them to
        // block_param_vregs_per_block would cause the allocator to treat them as
        // live-in (requiring a register at block entry), extending their live range
        // to all predecessors' exits and triggering spurious reloads.
        if !slot_spilled_params.is_empty() {
            for (&(bid, pidx), info) in &slot_spilled_params {
                let block_idx = block_id_to_idx[&bid];
                block_param_vregs_per_block[block_idx].remove(&info.vreg);
                if let Some(&ov) = block_param_vreg_overrides.get(&(bid, pidx)) {
                    block_param_vregs_per_block[block_idx].remove(&ov);
                }
            }
        }

        // Coalescing's copy pairs, read off the final schedules. Resolving each
        // argument's class through the function-wide map answers a per-block
        // question with whichever VReg the class was last given: a pure class is
        // re-emitted in every block that needs it, so the answer can be a VReg
        // defined in a block the edge never reaches, and merging the destination
        // parameter onto that VReg hands the parameter a register chosen for an
        // unrelated value.
        //
        // CRITICAL ORDER: after the splitter, so an argument it routed through a
        // stack slot -- which has no operand and no copy -- contributes no pair.
        let copy_pairs = compute_copy_pairs_from_schedules(
            func,
            &block_schedules,
            &egraph,
            &class_to_vreg,
            &block_param_map,
            &block_param_vreg_overrides,
            &block_param_vregs,
        );

        // `phi_uses` covers every VReg a terminator consumes -- a Jump's and a
        // Branch's arguments and a `Ret`'s value alike, since
        // `barrier::terminator_arg_classes` numbers all three the same way -- so
        // the allocator has the whole terminator half of liveness.
        verify_phi_uses = phi_uses.clone();
        verify_block_params = block_param_vregs_per_block.clone();
        verify_copy_pairs = copy_pairs.clone();
        let global_result = allocate_global(
            &block_schedules,
            &param_vregs,
            call_arg_precolors,
            &copy_pairs,
            &loop_depths,
            &cfg_succs,
            &phi_uses,
            &block_param_vregs_per_block,
            &func.name,
            opts.force_frame_pointer,
            &mut slots,
        )
        .map_err(|e| CompileError {
            phase: "regalloc".into(),
            message: e,
            location: Some(IrLocation {
                function: func.name.clone(),
                block: None,
                inst: None,
            }),
        })?;

        let block_rewritten_storage = global_result.per_block_insts;
        let merged_vreg_to_reg = global_result.vreg_to_reg;
        let mut merged_callee_saved = global_result.callee_saved_used;
        let global_unprecolored_params = global_result.unprecolored_params;
        let coalesce_aliases: BTreeMap<VReg, VReg> = global_result.coalesce_aliases;

        // Every slot in these schedules came from `slots`, whichever pass spilled
        // to it, so the frame reserves exactly what it handed out.
        let spill_slot_counter = slots.count();

        merged_callee_saved.sort_by_key(|r| *r as u8);
        merged_callee_saved.dedup();

        let merged_result = RegAllocResult {
            vreg_to_reg: merged_vreg_to_reg,
            spill_slots: spill_slot_counter,
            callee_saved_used: merged_callee_saved,
            insts: vec![],
            unprecolored_params: global_unprecolored_params,
        };

        (merged_result, block_rewritten_storage, coalesce_aliases)
    };

    // A spill op naming a slot no pass allocated addresses a frame cell the
    // prologue does not reserve, or one another value owns. Nothing downstream can
    // see it: the displacement is well-formed and the store writes it, so both the
    // machine verifier and the slot-traffic dump read it as an ordinary slot.
    if cfg!(debug_assertions) {
        let unowned: BTreeSet<i64> = block_rewritten
            .iter()
            .flatten()
            .filter_map(|inst| match inst.op {
                Op::SpillStore(slot)
                | Op::SpillLoad(slot)
                | Op::XmmSpillStore(slot)
                | Op::XmmSpillLoad(slot) => slots.owner(slot as u32).is_none().then_some(slot),
                _ => None,
            })
            .collect();
        assert!(
            unowned.is_empty(),
            "[{}] spill ops name slots {unowned:?}, outside the {} this function allocated",
            func.name,
            slots.count(),
        );
    }

    if let Some(s) = sink.as_mut() {
        s.phase_stats(
            "regalloc",
            &format!(
                "regs_used={}, spill_slots={}",
                regalloc_result.vreg_to_reg.len(),
                regalloc_result.spill_slots
            ),
        );
    }

    if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(&func.name) {
        tracing::debug!(
            target: "blitz::regalloc",
            "[{}] final assignment ({} vregs, {} spill slots):\n{}",
            func.name,
            regalloc_result.vreg_to_reg.len(),
            regalloc_result.spill_slots,
            crate::trace::format_vreg_to_reg(&regalloc_result.vreg_to_reg),
        );
    }

    // Two VRegs live at once must not share a register. Checked here rather than
    // at machine level because it needs the VRegs, which the rewrite ahead
    // erases, and liveness recomputed from the schedules as emitted -- asking the
    // allocator's own interference graph would be circular.
    if crate::verify::is_enabled() && func.blocks.len() > 1 {
        let succs = cfg::successor_indices(func);
        let errors = crate::verify::verify_register_sharing(
            &block_rewritten,
            &verify_phi_uses,
            &verify_block_params,
            &succs,
            &regalloc_result.vreg_to_reg,
            &coalesce_aliases,
            &verify_copy_pairs,
        );
        if !errors.is_empty() {
            panic!(
                "BLITZ_VERIFY: {} register-sharing violation(s) in function '{}' after \
                 register allocation:\n  - {}",
                errors.len(),
                func.name,
                errors.join("\n  - ")
            );
        }
    }

    // Build the set of param VRegs so lowering can skip their Iconst sentinels.
    let param_vreg_set: BTreeSet<VReg> = param_vregs.iter().map(|(v, _)| *v).collect();

    // Compute frame layout early so spill lowering can use it during Phase 7.
    let frame_layout = compute_frame_layout(
        regalloc_result.spill_slots,
        &regalloc_result.callee_saved_used,
        0,
        has_calls,
        opts.force_frame_pointer,
        user_stack_slots,
    );

    // Phase 7: Per-block MachInst lowering + phi elimination + terminator emission.
    // Blocks are processed and emitted in RPO order.
    // LabelIds are block IDs (block.id), which are stable across reordering.
    let n_blocks = func.blocks.len();
    // Extra labels for trampoline code start after the maximum block id + 1.
    let max_block_id = func.blocks.iter().map(|b| b.id).max().unwrap_or(0);
    let mut next_label: LabelId = max_block_id + 1;
    // block_items[i] holds the items for the block at rpo_order[i].
    let mut block_items: Vec<Vec<BlockItem>> = Vec::with_capacity(n_blocks);

    for (rpo_pos, &block_idx) in rpo_order.iter().enumerate() {
        let block = &func.blocks[block_idx];
        // Strip barrier pseudo-ops before Phase 7 grouping: their dummy dst VRegs
        // are not in barrier maps and would be misrouted to group 0.
        let rewritten: Vec<ScheduledInst> = block_rewritten[block_idx]
            .iter()
            .filter(|inst| !matches!(inst.op, Op::StoreBarrier | Op::VoidCallBarrier))
            .cloned()
            .collect();
        let rewritten = &rewritten;
        // Retain the un-stripped schedule for effectful-op lowering: the
        // StoreBarrier/VoidCallBarrier pseudo-ops carry the (post-spill)
        // operand renames that resolve_*_regs_after_spilling uses to find
        // reload/remat VRegs for Store val and Call args.
        let full_schedule_for_barriers = &block_rewritten[block_idx];

        // The block that follows this one in emission order (for fallthrough).
        let next_block_id: Option<BlockId> = rpo_order
            .get(rpo_pos + 1)
            .map(|&next_idx| func.blocks[next_idx].id);

        // Build a block-local class_to_vreg using the per-block snapshot.
        // Use the per-block snapshot (captured post-emission, pre-restore) so
        // classes re-emitted in this block resolve to THIS block's VReg — not
        // a stale one from a non-dominating prior block that was restored into
        // the global `class_to_vreg`.
        let block_class_to_vreg: ClassVRegMap = {
            let snapshot = &block_class_to_vreg_snapshot[block_idx];
            let mut map: ClassVRegMap = snapshot.clone();
            // Apply override VRegs for block params: update class_to_vreg mapping
            // so phi copy source lookups find the correct register.
            for (&(bid, pidx), &fresh_vreg) in &block_param_vreg_overrides {
                if let Some(&param_cid) = block_param_map.get(&(bid, pidx)) {
                    let canon = egraph.unionfind.find_immutable(param_cid);
                    if bid == block.id {
                        // This block defines the override VReg.
                        map.insert_single(canon, fresh_vreg);
                    }
                }
            }
            // Rename every VReg to the one that survived coalescing, keeping the
            // range it was recorded with. `apply_coalescing` renamed the
            // schedules and not this map, so a merged-away VReg has no register
            // assignment and reads as no answer at all.
            //
            // Renaming in place, rather than collapsing each class to a single
            // full-range VReg: the collapse discarded everything the splitter
            // recorded, so with a class holding a value plus a reload per block,
            // one segment won every lookup and a phi copy in one block read the
            // copy belonging to another. The full-range entries linearization
            // made are still here and still act as the fallback where a class
            // has no narrower segment covering the point; what is gone is
            // fabricating one.
            //
            // The chase is transitive. A single step leaves a VReg that Phase 3
            // merged twice still pointing at an intermediate with no register.
            if !coalesce_aliases.is_empty() {
                let mut aliased_map = ClassVRegMap::new();
                for (cid, vreg, start, end) in map.iter_segments() {
                    aliased_map.insert_segment_shared(
                        cid,
                        terminator::chase_alias(vreg, &coalesce_aliases),
                        start,
                        end,
                    );
                }
                map = aliased_map;
            }
            map
        };

        // Handle non-terminator effectful ops (loads, stores, calls).
        let non_term_count = block.non_term_count();
        let non_term_ops = &block.ops[..non_term_count];

        let mut all_insts: Vec<MachInst> = Vec::new();

        // Emit movs for register params not precolored (live across a call
        // that clobbers their ABI register). Must be at the very start of
        // the function, before any call arg setup.
        let arg_locs = &func_arg_locs;
        for inst in rewritten.iter() {
            if let Op::Param(param_idx, _) = &inst.op
                && !param_vreg_set.contains(&inst.dst)
                && let Some(crate::x86::abi::ArgLoc::Reg(abi_reg)) =
                    arg_locs.get(*param_idx as usize)
                && let Some(&dst_reg) = regalloc_result.vreg_to_reg.get(&inst.dst)
                && dst_reg != *abi_reg
            {
                all_insts.push(MachInst::MovRR {
                    size: crate::x86::inst::OpSize::S64,
                    src: crate::x86::inst::Operand::Reg(*abi_reg),
                    dst: crate::x86::inst::Operand::Reg(dst_reg),
                });
            }
        }
        // Emit entry movs for unprecolored params from the global
        // allocator. Only in the entry block; these are params whose ABI
        // precoloring was dropped by merge_precolorings_global because they
        // are live across a call that clobbers their ABI register.
        if block_idx == rpo_order[0] {
            for &(param_vreg, abi_reg) in &regalloc_result.unprecolored_params {
                if let Some(&dst_reg) = regalloc_result.vreg_to_reg.get(&param_vreg)
                    && dst_reg != abi_reg
                {
                    if abi_reg.is_xmm() {
                        all_insts.push(MachInst::MovsdRR {
                            dst: crate::x86::inst::Operand::Reg(dst_reg),
                            src: crate::x86::inst::Operand::Reg(abi_reg),
                        });
                    } else {
                        all_insts.push(MachInst::MovRR {
                            size: crate::x86::inst::OpSize::S64,
                            src: crate::x86::inst::Operand::Reg(abi_reg),
                            dst: crate::x86::inst::Operand::Reg(dst_reg),
                        });
                    }
                }
            }
        }

        // Emit in schedule order.
        //
        // The schedule was ordered by barrier group during linearization, before
        // register allocation, and the splitter inserts at positions inside it.
        // So this order is both the intended emission order and -- the part that
        // matters -- the order the allocator measured liveness on.
        //
        // Assigning barrier groups a second time here and emitting in *that*
        // answer is what let a register be clobbered between a def and its use:
        // wherever the two groupings disagreed, a value was emitted somewhere its
        // interference had never been measured, and no amount of pinning
        // individual VRegs back into place fixed the general case.
        //
        // A barrier instruction stands in for its effectful op. Emit the load,
        // store or call at the barrier's own position and no code for the
        // pseudo-op itself; everything else lowers as a pure op.
        // Computed over the whole block: lowering below runs on each run of pure
        // ops between barriers, and a projection is often in a different run from
        // the op it projects.
        let (div_dst_vregs, has_proj0_consumer) =
            crate::compile::lower::division_and_proj0_sets(full_schedule_for_barriers);

        let lower_pending = |pending: &mut Vec<ScheduledInst>,
                             out: &mut Vec<MachInst>|
         -> Result<(), CompileError> {
            if pending.is_empty() {
                return Ok(());
            }
            out.extend(lower_block_pure_ops(
                pending,
                &regalloc_result,
                func,
                &param_vreg_set,
                &frame_layout,
                &vreg_types,
                arg_locs,
                &div_dst_vregs,
                &has_proj0_consumer,
            )?);
            pending.clear();
            Ok(())
        };

        let mut pending: Vec<ScheduledInst> = Vec::new();
        let mut barrier_k = 0usize;
        for inst in full_schedule_for_barriers.iter() {
            if !matches!(
                inst.op,
                Op::CallResult(_, _)
                    | Op::LoadResult(_, _)
                    | Op::VoidCallBarrier
                    | Op::StoreBarrier
            ) {
                pending.push(inst.clone());
                continue;
            }
            lower_pending(&mut pending, &mut all_insts)?;
            // Barrier instructions appear in effectful-op order, which is what
            // makes counting them the barrier index.
            if let Some(op) = non_term_ops.get(barrier_k) {
                all_insts.extend(lower_effectful_op(
                    op,
                    block_idx,
                    barrier_k,
                    &block_class_to_vreg,
                    &regalloc_result,
                    func,
                    &egraph.unionfind,
                    full_schedule_for_barriers,
                )?);
            }
            barrier_k += 1;
        }
        lower_pending(&mut pending, &mut all_insts)?;
        debug_assert_eq!(
            barrier_k,
            non_term_ops.len(),
            "block {} of '{}' has {} barrier instructions for {} effectful ops",
            block_idx,
            func.name,
            barrier_k,
            non_term_ops.len(),
        );

        // After the global allocator, every VReg in the schedule must have a
        // physical register. Only active for multi-block functions (single-block
        // uses the old allocator path with its own guarantees).
        //
        // A Ret's value needs no check here: `lower_terminator` fails outright
        // when it cannot name a register for one, which is the same property
        // stated where the register is actually needed, and unconditionally
        // rather than only in a debug build.
        if func.blocks.len() > 1 {
            for inst in rewritten.iter() {
                for &op in &inst.operands {
                    debug_assert!(
                        regalloc_result.vreg_to_reg.contains_key(&op),
                        "8a-effectful safety net fired after global regalloc: \
                         operand VReg {:?} in block {} of function '{}' has no register assignment",
                        op,
                        block_idx,
                        func.name,
                    );
                }
            }
        }

        // Handle the terminator.
        let terminator = block.ops.last().expect("block must have terminator");
        let term_items = lower_terminator(
            terminator,
            block_idx,
            next_block_id,
            &egraph,
            &class_to_vreg,
            &block_class_to_vreg,
            &block_param_map,
            &block_param_vreg_overrides,
            &block_param_vregs,
            // Straight off the final schedule: post-split, post-coalesce, the
            // VRegs the allocator actually assigned registers to.
            &barrier::terminator_arg_operands(&block_rewritten[block_idx])
                .into_iter()
                .collect::<BTreeMap<u32, VReg>>(),
            &coalesce_aliases,
            &regalloc_result,
            func,
            &mut next_label,
            &slot_spilled_params,
            &frame_layout,
            &block_rewritten,
        )?;

        // Phase 8: Peephole on this block's pure/effectful instructions.
        let final_insts = if opts.enable_peephole {
            peephole(all_insts)
        } else {
            all_insts
        };

        // Reassemble into BlockItems.
        let mut items: Vec<BlockItem> = final_insts.into_iter().map(BlockItem::Inst).collect();
        items.extend(term_items);
        block_items.push(items);
    }

    // Branch threading: rewrite Jcc/Jmp targets that point to empty blocks
    // containing only a single Jmp. Repeat until fixed point.
    thread_branches(&mut block_items, func, &rpo_order);

    // Phase 10: Encoding with branch relaxation.
    //
    // Step 10a: Flatten all BlockItems into a linear instruction sequence,
    // recording label positions (label -> instruction index immediately after
    // the label binding point).
    //
    // Block labels are bound at the start of each block; trampoline labels
    // (BlockItem::BindLabel) are bound at whatever position they appear.
    // We represent label bindings as a sentinel NOP(0) to anchor their
    // position in the flat list, paired with a side table of label->inst_idx.

    // flat_insts: the instruction sequence passed to relax_branches.
    // flat_labels: for each instruction index, any labels bound just before it.
    // label_positions: label -> instruction index (for relax_branches).
    // Block labels use block.id (not block_idx) so Jump targets resolve correctly.
    let mut flat_insts: Vec<MachInst> = Vec::new();
    let mut label_positions: BTreeMap<LabelId, usize> = BTreeMap::new();

    for (rpo_pos, items) in block_items.iter().enumerate() {
        let block_id = func.blocks[rpo_order[rpo_pos]].id;
        // The block label is bound before the first instruction of this block.
        label_positions.insert(block_id as LabelId, flat_insts.len());

        for item in items {
            match item {
                BlockItem::Inst(inst) => {
                    flat_insts.push(inst.clone());
                }
                BlockItem::BindLabel(label_id) => {
                    // Trampoline label: bound at the position of the next instruction.
                    label_positions.insert(*label_id, flat_insts.len());
                }
            }
        }
    }

    // Step 10b: Branch relaxation -- determine which jumps use short (rel8) form.
    //
    // MachInst::Ret is lowered by `emit_epilogue`, not by `encode_inst`. The
    // default `inst_size` routes Ret through `encode_inst` (a single c3 byte)
    // and therefore underestimates the expansion to epilogue size (frame
    // teardown + callee-saved pops + ret). relax_branches' byte offsets would
    // drift, potentially leaving a short jump whose real displacement is out
    // of rel8 range (panics at fixup time). Provide a size oracle that
    // substitutes the actual epilogue byte count for each Ret.
    let epilogue_size = {
        let mut scratch = Encoder::new();
        emit_epilogue(&mut scratch, &frame_layout);
        scratch.buf.len()
    };
    let inst_size_for_relax = |inst: &MachInst| -> usize {
        if matches!(inst, MachInst::Ret) {
            epilogue_size
        } else {
            inst_size(inst)
        }
    };
    let (flat_insts, is_short) =
        crate::emit::relax::relax_branches(&flat_insts, &label_positions, &inst_size_for_relax);

    // Machine-level verification of the final stream: no surviving virtual
    // register, nothing reads a physical register the function never writes, and
    // no reload reads a spill slot nothing stored. This is the check that turns a
    // wrong-code bug from "segfault somewhere in a large program" into a named
    // instruction.
    if crate::trace::is_enabled("slots") && crate::trace::fn_matches(&func.name) {
        tracing::debug!(
            target: "blitz::slots",
            "[{}] spill slot traffic:\n{}",
            func.name,
            crate::trace::format_slot_traffic(
                &flat_insts,
                frame_layout.spill_base,
                frame_layout.spill_offset,
                &slots,
            ),
        );
    }

    crate::verify::verify_machinsts_stage(
        "encode",
        &func.name,
        &flat_insts,
        &label_positions,
        &frame_layout,
        regalloc_result.spill_slots,
    );

    // Step 10c: Encode.
    let mut encoder = Encoder::new();
    let func_start = encoder.buf.len();
    emit_prologue(&mut encoder, &frame_layout);

    // Bind block labels and trampoline labels in RPO order.
    // Labels use block.id so that jump targets encoded in lower_terminator resolve.
    let mut flat_idx = 0usize;
    for (rpo_pos, items) in block_items.iter().enumerate() {
        let block_id = func.blocks[rpo_order[rpo_pos]].id;
        encoder.bind_label(block_id as LabelId);

        for item in items {
            match item {
                BlockItem::Inst(inst) => {
                    let short = is_short[flat_idx];
                    flat_idx += 1;
                    if *inst == MachInst::Ret {
                        emit_epilogue(&mut encoder, &frame_layout);
                    } else {
                        encoder.encode_inst_with_form(&flat_insts[flat_idx - 1], short);
                    }
                }
                BlockItem::BindLabel(label_id) => {
                    encoder.bind_label(*label_id);
                }
            }
        }
    }

    encoder.resolve_fixups();

    let func_size = encoder.buf.len() - func_start;

    if let Some(s) = sink.as_mut() {
        s.phase_stats("encoding", &format!("bytes={func_size}"));
    }

    if crate::trace::is_enabled("asm") && crate::trace::fn_matches(&func.name) {
        let code_bytes = &encoder.buf[func_start..];
        if let Some(disasm) = crate::test_utils::objdump_disasm(code_bytes) {
            tracing::debug!(
                target: "blitz::asm",
                "[{}] disassembly ({func_size} bytes):\n{disasm}",
                func.name,
            );
        } else {
            tracing::debug!(
                target: "blitz::asm",
                "[{}] disassembly unavailable (objdump not found), {func_size} bytes",
                func.name,
            );
        }
    }

    // Collect externals (symbols referenced by call instructions).
    let externals: Vec<String> = collect_externals(func);

    Ok(ObjectFile {
        code: encoder.buf,
        relocations: encoder.relocations,
        functions: vec![FunctionInfo {
            name: func.name.clone(),
            offset: func_start,
            size: func_size,
        }],
        externals,
        globals: vec![],
        rodata: vec![],
    })
}

mod ir_print;
pub use ir_print::{compile_module_to_ir, compile_to_ir_string};

mod module;
pub use module::{compile_module, compile_module_with_globals};

/// A flat item emitted for a block: either a MachInst or a label binding.
pub(crate) enum BlockItem {
    Inst(MachInst),
    BindLabel(LabelId),
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
