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

use std::collections::{BTreeMap, BTreeSet, HashMap};

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
use crate::schedule::scheduler::{ScheduleDag, ScheduledInst, schedule};
use crate::x86::abi::{compute_frame_layout, emit_epilogue, emit_prologue};
use crate::x86::encode::{Encoder, inst_size};
use crate::x86::inst::{LabelId, MachInst};
use crate::x86::reg::Reg;

mod barrier;
pub mod program_point;
use barrier::{
    assign_barrier_groups, build_barrier_context, insert_early_barrier_spills,
    populate_effectful_operands,
};
use program_point::ProgramPoint;
mod cfg;
use cfg::{
    build_block_param_class_map, collect_block_roots, collect_externals, collect_phi_source_vregs,
    collect_roots, compute_copy_pairs, compute_idom, compute_loop_depths, compute_rpo, dominates,
};
mod effectful;
use effectful::lower_effectful_op;
mod dce;
mod licm;
mod lower;
use lower::lower_block_pure_ops;
mod precolor;
use precolor::{
    add_call_precolors_for_block, add_div_precolors, add_shift_precolors,
    assign_param_vregs_from_map,
};
mod terminator;
use terminator::{lower_terminator, thread_branches};
pub(crate) mod split;

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

    let block_param_map = build_block_param_class_map(egraph);

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

    // LICM: detect loops, insert preheaders, identify invariant classes.
    let extra_roots = if opts.enable_licm {
        licm::run_licm(&mut func, &mut egraph)
    } else {
        Default::default()
    };

    // Phases 1-2: E-graph rewrites and cost-based extraction.
    // Temporarily borrow func for extraction; DCE2 needs &mut func afterwards.
    let (block_param_map, extraction) = run_egraph_and_extract(&func, &mut egraph, opts)?;

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
    // Must run BEFORE the immutable reborrow and before index structures are built.
    let extra_roots = if opts.enable_dce {
        dce::run_dce2_with_extra_roots(&mut func, &egraph, &extraction, extra_roots)
    } else {
        extra_roots
    };

    // NOW freeze func for the rest of the pipeline.
    let func = &func;

    // Build BlockId -> index map for O(1) lookups (must be after DCE2).
    let block_id_to_idx: HashMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // Detect whether this function contains any call instructions (must be after DCE2).
    let has_calls = func_has_calls(func);

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
    let (block_preds, _) = licm::build_predecessor_map(func);

    // Map (BlockId, param_idx) -> fresh VReg for block params whose canonical
    // VReg was emitted by a prior block. This prevents the e-graph from merging
    // outer and inner loop header params into the same register.
    let mut block_param_vreg_overrides: BTreeMap<(BlockId, u32), VReg> = BTreeMap::new();

    let idom = compute_idom(func, &rpo_order);
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
    let mut vreg_types = build_vreg_types(&class_to_vreg, &egraph);

    // Insert types for fresh block param VRegs allocated above.
    for (&(bid, pidx), &fresh_vreg) in &block_param_vreg_overrides {
        let block = &func.blocks[block_id_to_idx[&bid]];
        let ty = block.param_types[pidx as usize].clone();
        vreg_types.insert(fresh_vreg, ty);
    }

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

    // Hoist Param ops to the front of the entry block's schedule.
    if !block_schedules.is_empty() {
        let sched = &mut block_schedules[0];
        sched.sort_by_key(|inst| {
            if matches!(inst.op, Op::Param(_, _)) {
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

    // Build phi copy pairs from block parameter passing for coalescing.
    let copy_pairs = compute_copy_pairs(
        func,
        &class_to_vreg,
        &egraph,
        &block_param_map,
        &block_param_vreg_overrides,
    );

    // Compute loop depths from the CFG for spill selection.
    let loop_depths = compute_loop_depths(func, &block_schedules);

    // Block params that are slot-spilled by the Phase 6 splitter.
    // Populated inside the multi-block path if BLITZ_SPLIT=1.
    // Passed to lower_terminator so predecessor terminators emit slot stores.
    let mut slot_spilled_params: crate::compile::split::BlockParamSlotMap =
        std::collections::BTreeMap::new();

    // Single-block fast path skips global liveness.
    let (regalloc_result, block_rewritten, block_rename_maps, coalesce_aliases) = if func
        .blocks
        .len()
        == 1
    {
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
        collect_phi_source_vregs(func, &egraph, &class_to_vreg, &mut live_out);
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
        let result = allocate(
            &all_scheduled,
            &all_param_vregs,
            &live_out,
            &copy_pairs,
            &loop_depths,
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
        let rename_maps: Vec<BTreeMap<VReg, VReg>> = vec![BTreeMap::new()];
        let aliases: BTreeMap<VReg, VReg> = BTreeMap::new();
        (result, rewritten, rename_maps, aliases)
    } else {
        // --- Multi-block path: global register allocator (Phase 6 cutover) ---

        // Step 1: Compute CFG successors and phi uses per block.
        let cfg_succs = crate::regalloc::global_liveness::cfg_successors(func);
        let mut phi_uses = crate::regalloc::global_liveness::compute_phi_uses(
            func,
            &egraph.unionfind,
            &class_to_vreg,
        );

        // Post-process phi_uses: replace back-edge terminator VRegs with their
        // block-param override VRegs so cross-block liveness is correct.
        crate::regalloc::global_liveness::apply_block_param_overrides_to_phi_uses(
            func,
            &egraph.unionfind,
            &block_param_vreg_overrides,
            &block_param_map,
            &class_to_vreg,
            &rpo_order,
            &mut phi_uses,
        );

        // Task 6.1a (CRITICAL ORDER): Collect call-arg precolors BEFORE calling
        // populate_effectful_operands. The barrier system sorts operands by VReg
        // index (barrier.rs:388,405-406), destroying ABI argument order.
        // add_call_precolors_for_block reads EffectfulOp::Call args in positional
        // ABI order and must run BEFORE the barrier sort.
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

        // Task 6.4: insert_early_barrier_spills per block before allocate_global.
        // Tracks slot numbers used so we can separate them from global-allocator
        // slots after the fact (the global allocator also starts its slot counter
        // from 0 internally, so we need to distinguish the two ranges).
        let mut early_barrier_slots: std::collections::BTreeSet<u32> =
            std::collections::BTreeSet::new();
        let mut pre_spill_slots: u32 = 0;
        {
            let mut early_next_vreg = next_vreg;
            for (block_idx, block) in func.blocks.iter().enumerate() {
                let non_term_count = block.non_term_count();
                if non_term_count > 0 {
                    let (result_map, arg_map) =
                        build_barrier_context(block, block_idx, &egraph, &class_to_vreg);
                    let mut vreg_group =
                        assign_barrier_groups(&block_schedules[block_idx], &result_map, &arg_map);
                    let slots_before = pre_spill_slots;
                    insert_early_barrier_spills(
                        &mut block_schedules[block_idx],
                        &result_map,
                        &arg_map,
                        &mut vreg_group,
                        &vreg_types,
                        &mut early_next_vreg,
                        &mut pre_spill_slots,
                    );
                    // Record which slot numbers were allocated by early-barrier spills.
                    for slot in slots_before..pre_spill_slots {
                        early_barrier_slots.insert(slot);
                    }
                }
            }
            next_vreg = early_next_vreg;
        }

        // Populate effectful-op operands onto barrier instructions (LoadResult,
        // CallResult, StoreBarrier, VoidCallBarrier) in each block's schedule
        // BEFORE global liveness, so compute_global_liveness sees them as regular
        // instruction operands and includes them in cross-block liveness.
        // This MUST happen AFTER call_arg_precolors collection (Task 6.1a).
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let non_term_count = block.non_term_count();
            if non_term_count > 0 {
                let non_term_ops = &block.ops[..non_term_count];
                let (result_map, arg_map) =
                    build_barrier_context(block, block_idx, &egraph, &class_to_vreg);
                let mut vreg_group =
                    assign_barrier_groups(&block_schedules[block_idx], &result_map, &arg_map);
                populate_effectful_operands(
                    &mut block_schedules[block_idx],
                    non_term_ops,
                    block_idx,
                    &egraph,
                    &class_to_vreg,
                    &mut vreg_group,
                    &mut next_vreg,
                );
            }
        }

        // Pressure-driven splitter (env-gated: BLITZ_SPLIT=1).
        // CRITICAL ORDER: apply_plan_to must run BEFORE collect_block_param_vregs_per_block.
        // The splitter may truncate segments, which affects what block params are found.
        //
        if std::env::var("BLITZ_SPLIT").ok().as_deref() == Some("1") {
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
            let mut plan = split::plan_splits(
                &block_schedules,
                &class_to_vreg,
                &extraction,
                &egraph,
                &split_cost_model,
                &split_global_liveness,
                gpr_budget,
                xmm_budget,
                next_vreg,
                &loop_depths,
                func,
            );
            // Extract slot_spilled_params before consuming the plan.
            slot_spilled_params = std::mem::take(&mut plan.slot_spilled_params);

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
                {
                    if call_arg_vreg_set.contains(&old_vreg) {
                        // Only keep first entry (the call-site reload VReg; later
                        // entries for the same old VReg are non-call-site reloads).
                        vreg_remap.entry(old_vreg).or_insert(new_vreg);
                    }
                }
            }

            split::apply_plan_to(
                &mut block_schedules,
                &mut class_to_vreg,
                &mut next_vreg,
                plan,
            );

            // Update call_arg_precolors: transfer each precolor to its reload VReg.
            if !vreg_remap.is_empty() {
                for (precolor_vreg, _reg) in call_arg_precolors.iter_mut() {
                    if let Some(&new_vreg) = vreg_remap.get(precolor_vreg) {
                        *precolor_vreg = new_vreg;
                    }
                }
            }

            // Remove slot-spilled param VRegs from phi_uses.
            // phi_uses was computed before apply_plan_to so it still references
            // the original param VRegs (and their block-param overrides after
            // apply_block_param_overrides_to_phi_uses). After slot-spilling,
            // these VRegs have no registers; the allocator would try to reload
            // them at block exit, creating spurious reloads that overwrite the
            // slot with wrong values. The phi copy for a slot-spilled param is
            // emitted as a slot store (PhiCopy::Slot) by lower_terminator.
            for (&(bid, pidx), info) in &slot_spilled_params {
                // Remove both the original VReg and any block-param override VReg
                // (apply_block_param_overrides_to_phi_uses may have replaced it).
                let override_vreg = block_param_vreg_overrides.get(&(bid, pidx)).copied();
                for phi_set in phi_uses.iter_mut() {
                    phi_set.remove(&info.vreg);
                    if let Some(ov) = override_vreg {
                        phi_set.remove(&ov);
                    }
                }
            }
        } else {
            // Splitter not running: bump split_generation so that
            // collect_block_param_vregs_per_block's ordering assert passes.
            class_to_vreg.split_generation += 1;
        }

        // Step 3: Determine block params per block (passed to allocate_global).
        // CRITICAL ORDER: must run AFTER apply_plan_to (splitter output committed).
        let mut block_param_vregs_per_block =
            crate::regalloc::global_liveness::collect_block_param_vregs_per_block(
                func,
                &egraph,
                &class_to_vreg,
            );

        // Include fresh block param VRegs in the per-block sets.
        for (&(bid, _pidx), &fresh_vreg) in &block_param_vreg_overrides {
            let block_idx = block_id_to_idx[&bid];
            block_param_vregs_per_block[block_idx].insert(fresh_vreg);
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

        // Task 6.1 / Task 6.2: Call allocate_global. This replaces the entire
        // per-block loop (assign_cross_block_slots + rewrite_block_for_splitting +
        // allocate() per block). Those functions are no longer called here.
        // `phi_uses` already includes Ret values (compute_phi_uses covers Ret
        // val, Jump args, Branch args), so the allocator has the full set of
        // terminator-consumed VRegs for liveness + end-of-block reload logic.
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

        // Task 6.5 (deleted per-block spill slot offset code): no per-block slot
        // offsetting is needed. The global allocator uses a single slot space 0..M.
        // Early-barrier spills (pre_spill_slots) are in the input schedule with
        // slot numbers 0..pre_spill_slots. The global allocator's new spills use
        // 0..M internally. We shift the global-allocator slots by +pre_spill_slots
        // and leave early-barrier slots unchanged, giving disjoint ranges.
        let mut block_rewritten_storage = global_result.per_block_insts;
        let block_rename_maps = global_result.per_block_rename_maps;
        let merged_vreg_to_reg = global_result.vreg_to_reg;
        let mut merged_callee_saved = global_result.callee_saved_used;
        let global_alloc_slots = global_result.spill_slots;
        let global_unprecolored_params = global_result.unprecolored_params;
        let coalesce_aliases: BTreeMap<VReg, VReg> = global_result.coalesce_aliases;

        // Shift global-allocator slot numbers by pre_spill_slots to avoid collision
        // with early-barrier slots (which occupy 0..pre_spill_slots).
        if pre_spill_slots > 0 {
            for block_insts in block_rewritten_storage.iter_mut() {
                for inst in block_insts.iter_mut() {
                    match &mut inst.op {
                        Op::SpillStore(slot)
                        | Op::SpillLoad(slot)
                        | Op::XmmSpillStore(slot)
                        | Op::XmmSpillLoad(slot) => {
                            // Shift global-allocator slots (not early-barrier ones).
                            // Early-barrier slots are in early_barrier_slots set.
                            // Note: after coalescing, early-barrier slot numbers in
                            // the result match the original 0..pre_spill_slots range.
                            if !early_barrier_slots.contains(&(*slot as u32)) {
                                *slot += pre_spill_slots as i64;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        let spill_slot_counter = global_alloc_slots + pre_spill_slots;

        merged_callee_saved.sort_by_key(|r| *r as u8);
        merged_callee_saved.dedup();

        let merged_result = RegAllocResult {
            vreg_to_reg: merged_vreg_to_reg,
            spill_slots: spill_slot_counter,
            callee_saved_used: merged_callee_saved,
            insts: vec![],
            unprecolored_params: global_unprecolored_params,
        };

        (
            merged_result,
            block_rewritten_storage,
            block_rename_maps,
            coalesce_aliases,
        )
    };

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

        // Build a block-local class_to_vreg that applies cross-block renames.
        // When a VReg was renamed (SpillLoad/remat) in this block, effectful ops
        // must use the renamed VReg to get the correct physical register.
        //
        // Use the per-block snapshot (captured post-emission, pre-restore) so
        // classes re-emitted in this block resolve to THIS block's VReg — not
        // a stale one from a non-dominating prior block that was restored into
        // the global `class_to_vreg`.
        let block_class_to_vreg: ClassVRegMap = {
            let snapshot = &block_class_to_vreg_snapshot[block_idx];
            let renames = &block_rename_maps[block_idx];
            let mut map: ClassVRegMap = if renames.is_empty() {
                snapshot.clone()
            } else {
                let mut m = ClassVRegMap::new();
                for (cid, vreg) in snapshot.iter() {
                    let renamed = renames.get(&vreg).copied().unwrap_or(vreg);
                    m.insert_single(cid, renamed);
                }
                m
            };
            // Apply override VRegs: if an override's fresh VReg was renamed
            // (reloaded) in this block, or this IS the block that defines it,
            // update the class_to_vreg mapping so phi copy source lookups
            // find the correct register.
            for (&(bid, pidx), &fresh_vreg) in &block_param_vreg_overrides {
                if let Some(&param_cid) = block_param_map.get(&(bid, pidx)) {
                    let canon = egraph.unionfind.find_immutable(param_cid);
                    if bid == block.id {
                        // This block defines the override VReg.
                        map.insert_single(canon, fresh_vreg);
                    } else if let Some(&renamed) = renames.get(&fresh_vreg) {
                        // The override was reloaded into this block.
                        map.insert_single(canon, renamed);
                    }
                }
            }
            // After renames and overrides, apply the global coalesce alias map
            // so `class_to_vreg[canon]` never points at a stale pre-coalesce
            // VReg that has no register assignment. The alias map is from
            // `allocate_global`'s result and is constant across blocks.
            if !coalesce_aliases.is_empty() {
                let mut aliased_map = ClassVRegMap::new();
                for (cid, vreg) in map.iter() {
                    let aliased = coalesce_aliases.get(&vreg).copied().unwrap_or(vreg);
                    aliased_map.insert_single(cid, aliased);
                }
                map = aliased_map;
            }
            map
        };

        // Handle non-terminator effectful ops (loads, stores, calls).
        let non_term_count = block.non_term_count();
        let non_term_ops = &block.ops[..non_term_count];

        // Build barrier maps using block_class_to_vreg (with cross-block renames
        // applied) so barrier operand caps match the renamed VRegs in the schedule.
        let (vreg_to_result_of_barrier, mut vreg_to_arg_of_barrier) =
            build_barrier_context(block, block_idx, &egraph, &block_class_to_vreg);

        // After spilling, CallResult/VoidCallBarrier operands may be SpillLoad
        // vregs that aren't in the original barrier arg map. Scan the schedule
        // and add them so barrier group assignment places them before the call.
        for inst in rewritten.iter() {
            if let Some(&barrier_k) = vreg_to_result_of_barrier.get(&inst.dst) {
                // This is a CallResult/LoadResult at barrier_k.
                // Its operands (possibly SpillLoad vregs) should be ready at barrier_k.
                for &op in &inst.operands {
                    let entry = vreg_to_arg_of_barrier.entry(op).or_insert(barrier_k);
                    *entry = (*entry).min(barrier_k);
                }
            }
            if matches!(inst.op, Op::VoidCallBarrier | Op::StoreBarrier) {
                // Find which barrier this belongs to by checking vreg_to_result_of_barrier.
                // VoidCallBarrier/StoreBarrier vregs aren't in vreg_to_result_of_barrier,
                // so check vreg_to_arg_of_barrier for operands.
                for &op in &inst.operands {
                    if let Some(&barrier_k) = vreg_to_arg_of_barrier.get(&inst.dst) {
                        let entry = vreg_to_arg_of_barrier.entry(op).or_insert(barrier_k);
                        *entry = (*entry).min(barrier_k);
                    }
                }
            }
        }

        let num_barriers = non_term_ops.len();

        // Partition scheduled pure insts into groups relative to barriers.
        let vreg_group = assign_barrier_groups(
            rewritten,
            &vreg_to_result_of_barrier,
            &vreg_to_arg_of_barrier,
        );
        let mut groups: Vec<Vec<&ScheduledInst>> = vec![Vec::new(); num_barriers + 1];
        for inst in rewritten.iter() {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            groups[g].push(inst);
        }
        // Emit pure ops interleaved with effectful ops.
        //
        // For each barrier K: emit group[K] pure ops, then barrier K.
        // After all barriers: emit group[num_barriers] (trailing pure ops).
        let mut all_insts: Vec<MachInst> = Vec::new();

        // Emit movs for register params not precolored (live across a call
        // that clobbers their ABI register). Must be at the very start of
        // the function, before any barrier group / call arg setup.
        let arg_locs = &func_arg_locs;
        for inst in rewritten.iter() {
            if let Op::Param(param_idx, _) = &inst.op {
                if !param_vreg_set.contains(&inst.dst) {
                    if let Some(crate::x86::abi::ArgLoc::Reg(abi_reg)) =
                        arg_locs.get(*param_idx as usize)
                    {
                        if let Some(&dst_reg) = regalloc_result.vreg_to_reg.get(&inst.dst) {
                            if dst_reg != *abi_reg {
                                all_insts.push(MachInst::MovRR {
                                    size: crate::x86::inst::OpSize::S64,
                                    src: crate::x86::inst::Operand::Reg(*abi_reg),
                                    dst: crate::x86::inst::Operand::Reg(dst_reg),
                                });
                            }
                        }
                    }
                }
            }
        }
        // Task 6.6: Emit entry movs for unprecolored params from the global
        // allocator. Only in the entry block; these are params whose ABI
        // precoloring was dropped by merge_precolorings_global because they
        // are live across a call that clobbers their ABI register.
        if block_idx == rpo_order[0] {
            for &(param_vreg, abi_reg) in &regalloc_result.unprecolored_params {
                if let Some(&dst_reg) = regalloc_result.vreg_to_reg.get(&param_vreg) {
                    if dst_reg != abi_reg {
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
        }

        let lower_group = |group: &[&ScheduledInst],
                           regalloc_result: &RegAllocResult,
                           func: &Function,
                           param_vreg_set: &BTreeSet<VReg>,
                           frame_layout: &crate::x86::abi::FrameLayout,
                           vreg_types: &BTreeMap<VReg, Type>|
         -> Result<Vec<MachInst>, CompileError> {
            lower_block_pure_ops(
                &group.iter().map(|&i| i.clone()).collect::<Vec<_>>(),
                regalloc_result,
                func,
                param_vreg_set,
                frame_layout,
                vreg_types,
                &arg_locs,
            )
        };

        for (barrier_k, op) in non_term_ops.iter().enumerate() {
            // Emit pure ops for group[barrier_k] before this barrier.
            let pre_insts = lower_group(
                &groups[barrier_k],
                &regalloc_result,
                func,
                &param_vreg_set,
                &frame_layout,
                &vreg_types,
            )?;
            all_insts.extend(pre_insts);
            // Emit the barrier (load/store/call).
            let extra = lower_effectful_op(
                op,
                block_idx,
                &block_class_to_vreg,
                &regalloc_result,
                &extraction,
                func,
                &egraph.unionfind,
                full_schedule_for_barriers,
            )?;
            all_insts.extend(extra);
        }
        // Emit trailing pure ops (group[num_barriers]).
        // When num_barriers == 0 this is group[0] containing all pure ops.
        // When num_barriers > 0 this is the group after the last barrier.
        {
            let post_insts = lower_group(
                &groups[num_barriers],
                &regalloc_result,
                func,
                &param_vreg_set,
                &frame_layout,
                &vreg_types,
            )?;
            all_insts.extend(post_insts);
        }

        // Task 6.3 debug asserts: after the global allocator cutover, every
        // VReg that appears in a terminator (Ret value) must have a physical
        // register in vreg_to_reg. These asserts fire if the global allocator's
        // insert_spills_global is incomplete. Only active for multi-block functions
        // (single-block uses the old allocator path with its own guarantees).
        if func.blocks.len() > 1 {
            let terminator_check = block.ops.last().expect("block must have terminator");
            // 8a-ret: Ret value must have a register.
            if let EffectfulOp::Ret { val: Some(cid) } = terminator_check {
                let canon = egraph.unionfind.find_immutable(*cid);
                if let Some(vreg) =
                    block_class_to_vreg.lookup(canon, ProgramPoint::block_exit(block_idx))
                {
                    debug_assert!(
                        regalloc_result.vreg_to_reg.contains_key(&vreg),
                        "8a-ret safety net fired after global regalloc: \
                         Ret value VReg {:?} has no register assignment in function '{}'",
                        vreg,
                        func.name,
                    );
                }
            }
            // 8a-phi and 8a-effectful: all VRegs in the rewritten schedule must
            // have register assignments (guaranteed by allocate_global).
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
            &coalesce_aliases,
            &regalloc_result,
            func,
            &mut next_label,
            &slot_spilled_params,
            &frame_layout,
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
