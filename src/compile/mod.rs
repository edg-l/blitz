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
use crate::egraph::extract::{VReg, VRegInst, build_vreg_types, extract, vreg_insts_for_block};
use crate::egraph::phases::{CompileOptions as EGraphOptions, run_phases};
use crate::emit::object::{FunctionInfo, ObjectFile};
use crate::emit::peephole::peephole;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::regalloc::allocator::{RegAllocResult, allocate};
use crate::regalloc::rewrite::rewrite_vregs;
use crate::schedule::scheduler::{ScheduleDag, ScheduledInst, schedule};
use crate::x86::abi::{compute_frame_layout, emit_epilogue, emit_prologue};
use crate::x86::encode::{Encoder, inst_size};
use crate::x86::inst::{LabelId, MachInst};
use crate::x86::reg::Reg;

mod barrier;
use barrier::{
    assign_barrier_groups, build_barrier_maps, insert_early_barrier_spills,
    mark_branch_cond_barrier, populate_effectful_operands,
};
mod cfg;
use cfg::{
    build_block_param_class_map, collect_block_roots, collect_externals, collect_phi_source_vregs,
    collect_roots, compute_copy_pairs, compute_idom, compute_loop_depths, compute_rpo, dominates,
};
mod effectful;
use effectful::lower_effectful_op;
mod lower;
use lower::lower_block_pure_ops;
mod precolor;
use precolor::{
    add_call_precolors_for_block, add_div_precolors, add_shift_precolors,
    assign_param_vregs_from_map, collect_call_points_for_block, collect_div_clobber_points,
};
mod terminator;
use terminator::{lower_terminator, thread_branches};

// ── Public options / error types ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub opt_goal: OptGoal,
    pub phase1_limit: u32,
    pub phase2_limit: u32,
    pub phase3_limit: u32,
    pub enable_peephole: bool,
    pub enable_nop_alignment: bool,
    pub verbosity: Verbosity,
    /// Force the frame pointer (push rbp / mov rbp, rsp / pop rbp) to always be emitted.
    /// Defaults to `false`: the frame pointer is omitted when not needed, freeing RBP as a
    /// general-purpose register. Set to `true` for debuggability or when a frame pointer is
    /// required (e.g. kernel code).
    pub force_frame_pointer: bool,
    /// Enable function inlining before optimization.
    pub enable_inlining: bool,
    /// Maximum inlining depth (transitive inlining limit).
    pub max_inline_depth: u32,
    /// Maximum callee e-graph node count to be considered for inlining.
    pub max_inline_nodes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Silent,
    Normal,
    Verbose,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            opt_goal: OptGoal::Balanced,
            phase1_limit: 10,
            phase2_limit: 5,
            phase3_limit: 5,
            enable_peephole: true,
            enable_nop_alignment: false,
            verbosity: Verbosity::Silent,
            force_frame_pointer: false,
            enable_inlining: false,
            max_inline_depth: 3,
            max_inline_nodes: 50,
        }
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
    let func = &func;

    // Detect whether this function contains any call instructions.
    // This is needed for frame layout decisions (leaf detection, red zone eligibility).
    let has_calls = func_has_calls(func);

    // Phase 1: E-graph rewrite rules.
    let egraph_opts = EGraphOptions {
        phase1_limit: opts.phase1_limit,
        phase2_limit: opts.phase2_limit,
        phase3_limit: opts.phase3_limit,
        max_classes: 500_000,
    };
    crate::egraph::algebraic::propagate_block_params(func, &mut egraph);
    run_phases(&mut egraph, &egraph_opts).map_err(|e| CompileError {
        phase: "egraph".into(),
        message: e,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;
    // Second pass catches constants revealed by folding in the first pass,
    // then re-run algebraic rules to fold newly exposed constant expressions.
    crate::egraph::algebraic::propagate_block_params(func, &mut egraph);
    crate::egraph::algebraic::apply_algebraic_rules(&mut egraph);
    egraph.rebuild();

    if let Some(s) = sink.as_mut() {
        s.phase_stats(
            "egraph",
            &format!(
                "classes={}, nodes={}",
                egraph.class_count(),
                egraph.node_count()
            ),
        );
    }

    // Build the block param class map (needed for phi copy generation and extraction roots).
    let block_param_map = build_block_param_class_map(&egraph);

    // Collect all root ClassIds from effectful ops + block params.
    // Block params must be roots so that continuation block params created
    // by inlining (which may not be reachable from any effectful op) still
    // get extracted and assigned VRegs.
    let mut all_roots = collect_roots(func);
    all_roots.extend(block_param_map.values().copied());
    all_roots.sort_by_key(|c| c.0);
    all_roots.dedup();

    // Phase 2: Extraction (shared across all blocks).
    let cost_model = CostModel::new(opts.opt_goal);
    let extraction = extract(&egraph, &all_roots, &cost_model).map_err(|e| CompileError {
        phase: "extraction".into(),
        message: e.to_string(),
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

    if let Some(s) = sink.as_mut() {
        s.phase_stats(
            "extraction",
            &format!("classes_extracted={}", extraction.choices.len()),
        );
    }

    // Phase 3: Build per-block VRegInst lists with a shared class_to_vreg map.
    //
    // We process blocks in RPO order so that loop headers come before loop
    // bodies and dominant definitions are visited before their uses.
    // Classes shared between blocks are only emitted by the first block that
    // reaches them (DFS deduplication).
    // DO NOT pre-populate class_to_vreg here — let the DFS assign VRegs
    // naturally so that param/block-param VRegInsts appear in the scheduled
    // list and regalloc can see them.
    let mut class_to_vreg: BTreeMap<ClassId, VReg> = BTreeMap::new();
    let mut next_vreg: u32 = 0;

    // Compute RPO block ordering (indices into func.blocks).
    let rpo_order = compute_rpo(func);

    // Map (BlockId, param_idx) -> fresh VReg for block params whose canonical
    // VReg was emitted by a prior block. This prevents the e-graph from merging
    // outer and inner loop header params into the same register.
    let mut block_param_vreg_overrides: BTreeMap<(BlockId, u32), VReg> = BTreeMap::new();

    let idom = compute_idom(func, &rpo_order);
    let mut class_emitted_in: BTreeMap<ClassId, usize> = BTreeMap::new();

    // Build per-block VRegInst lists in RPO order, stored by block index.
    let mut block_vreg_insts: Vec<Vec<VRegInst>> = vec![Vec::new(); func.blocks.len()];
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
            if let Some(vreg) = class_to_vreg.remove(&cid) {
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
        all_roots.sort_by_key(|c| c.0);
        all_roots.dedup();
        let pre_emission: BTreeSet<ClassId> = class_to_vreg.keys().copied().collect();
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
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    if let Some(inst) = insts.iter_mut().find(|i| i.dst == vreg) {
                        inst.op = Op::BlockParam(
                            block_id,
                            pidx,
                            block.param_types[pidx as usize].clone(),
                        );
                        inst.operands.clear();
                    } else {
                        // The VReg was emitted by a prior block. Allocate a
                        // fresh VReg local to this block to avoid the e-graph
                        // merging outer/inner loop header params.
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
        for cid in class_to_vreg.keys().copied().collect::<Vec<_>>() {
            if !pre_emission.contains(&cid) && !class_emitted_in.contains_key(&cid) {
                class_emitted_in.insert(cid, block_idx);
            }
        }

        // Restore removed classes so subsequent blocks can see them.
        for (cid, vreg) in removed {
            class_to_vreg.insert(cid, vreg);
        }
    }

    // Build VReg -> Type map from the egraph's per-class type info.
    let mut vreg_types = build_vreg_types(&class_to_vreg, &egraph);

    // Insert types for fresh block param VRegs allocated above.
    for (&(bid, pidx), &fresh_vreg) in &block_param_vreg_overrides {
        let block = func.blocks.iter().find(|b| b.id == bid).unwrap();
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
        let non_term_count = if block.ops.is_empty() {
            0
        } else {
            block.ops.len() - 1
        };
        if non_term_count == 0 {
            continue; // No effectful ops to constrain ordering.
        }
        let non_term_ops = &block.ops[..non_term_count];

        // Build barrier maps: which VRegs are produced/consumed by each barrier.
        let (vreg_to_result_of_barrier, mut vreg_to_arg_of_barrier) =
            build_barrier_maps(non_term_ops, &egraph, &class_to_vreg);
        mark_branch_cond_barrier(
            block.ops.last(),
            non_term_count,
            &egraph,
            &class_to_vreg,
            &mut vreg_to_arg_of_barrier,
        );

        let sched = &block_schedules[block_idx];
        let vreg_group =
            assign_barrier_groups(sched, &vreg_to_result_of_barrier, &vreg_to_arg_of_barrier);

        // Stable-sort by group to reorder while preserving within-group order.
        // Barrier results (LoadResult/CallResult) sort to the FRONT of their
        // group: their values are produced by effectful ops at the group
        // boundary, so the register is occupied from the start of the group.
        // Placing them after pure ops would let the regalloc think the register
        // is free, causing incorrect reuse and clobbering.
        let mut indexed: Vec<(usize, &ScheduledInst)> = sched.iter().enumerate().collect();
        indexed.sort_by_key(|(orig_idx, inst)| {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            let param_order: u8 = match inst.op {
                Op::Param(_, _) => 0,
                Op::LoadResult(_, _) | Op::CallResult(_, _) => 1,
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
    let param_vregs = assign_param_vregs_from_map(func, &class_to_vreg, &egraph, entry_has_calls);

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

    // Shared next_vreg counter for fresh VReg allocation across all blocks.
    let shared_next_vreg_start: u32 = block_schedules
        .iter()
        .flatten()
        .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    // Single-block fast path skips global liveness.
    let (regalloc_result, block_rewritten, block_rename_maps) = if func.blocks.len() == 1 {
        // --- Single-block fast path ---
        let mut all_scheduled: Vec<ScheduledInst> =
            block_schedules.iter().flatten().cloned().collect();

        // Populate effectful-op operands right before regalloc so it sees
        // effectful op operand liveness at the correct barrier positions.
        {
            let block = &func.blocks[0];
            let non_term_count = if block.ops.is_empty() {
                0
            } else {
                block.ops.len() - 1
            };
            if non_term_count > 0 {
                let non_term_ops = &block.ops[..non_term_count];
                let (result_map, mut arg_map) =
                    build_barrier_maps(non_term_ops, &egraph, &class_to_vreg);
                mark_branch_cond_barrier(
                    block.ops.last(),
                    non_term_count,
                    &egraph,
                    &class_to_vreg,
                    &mut arg_map,
                );
                let mut vreg_group = assign_barrier_groups(&all_scheduled, &result_map, &arg_map);
                populate_effectful_operands(
                    &mut all_scheduled,
                    non_term_ops,
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
            if let Some(&vreg) = class_to_vreg.get(&canon) {
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
            &egraph,
            &class_to_vreg,
            &mut all_param_vregs,
            &mut live_out,
        );
        let call_points =
            collect_call_points_for_block(func, 0, &all_scheduled, &class_to_vreg, &egraph);
        let div_points = collect_div_clobber_points(&all_scheduled);

        let result = allocate(
            &all_scheduled,
            &all_param_vregs,
            &live_out,
            &copy_pairs,
            &loop_depths,
            &call_points,
            &div_points,
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

        let rewritten = vec![rewrite_vregs(&result.insts, &result.vreg_to_reg)];
        let rename_maps: Vec<BTreeMap<VReg, VReg>> = vec![BTreeMap::new()];
        (result, rewritten, rename_maps)
    } else {
        // --- Multi-block path ---

        // Step 1: Compute CFG successors and phi uses per block.
        let cfg_succs = crate::regalloc::global_liveness::cfg_successors(func);
        let mut phi_uses = crate::regalloc::global_liveness::compute_phi_uses(
            func,
            &egraph.unionfind,
            &class_to_vreg,
        );

        // Post-process phi_uses: when a back-edge terminator arg's e-class
        // matches a target block param that has an override, replace the
        // global VReg with the override VReg so cross-block liveness keeps
        // it alive. Only apply on back edges (source RPO position >= target)
        // because forward edges should use the original VReg.
        let rpo_pos: BTreeMap<BlockId, usize> = rpo_order
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (func.blocks[idx].id, pos))
            .collect();
        for (block_idx, block) in func.blocks.iter().enumerate() {
            if let Some(term) = block.ops.last() {
                let src_pos = rpo_pos.get(&block.id).copied().unwrap_or(0);
                let mut process_args = |target: BlockId, args: &[ClassId]| {
                    let tgt_pos = rpo_pos.get(&target).copied().unwrap_or(0);
                    if src_pos < tgt_pos {
                        return; // Forward edge: use the original VReg.
                    }
                    for (pidx, &arg_cid) in args.iter().enumerate() {
                        if let Some(&fresh_vreg) =
                            block_param_vreg_overrides.get(&(target, pidx as u32))
                            && let Some(&param_cid) = block_param_map.get(&(target, pidx as u32))
                        {
                            let canon_arg = egraph.unionfind.find_immutable(arg_cid);
                            let canon_param = egraph.unionfind.find_immutable(param_cid);
                            if canon_arg == canon_param {
                                // Replace the global VReg with the override.
                                if let Some(&old_vreg) = class_to_vreg.get(&canon_arg) {
                                    phi_uses[block_idx].remove(&old_vreg);
                                }
                                phi_uses[block_idx].insert(fresh_vreg);
                            }
                        }
                    }
                };
                match term {
                    EffectfulOp::Jump { target, args } => {
                        process_args(*target, args);
                    }
                    EffectfulOp::Branch {
                        bb_true,
                        bb_false,
                        true_args,
                        false_args,
                        ..
                    } => {
                        process_args(*bb_true, true_args);
                        process_args(*bb_false, false_args);
                    }
                    _ => {}
                }
            }
        }

        // Populate effectful-op operands onto barrier instructions (LoadResult,
        // CallResult, StoreBarrier, VoidCallBarrier) in each block's schedule
        // BEFORE global liveness, so compute_global_liveness sees them as regular
        // instruction operands and includes them in cross-block liveness.
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let non_term_count = if block.ops.is_empty() {
                0
            } else {
                block.ops.len() - 1
            };
            if non_term_count > 0 {
                let non_term_ops = &block.ops[..non_term_count];
                let (result_map, mut arg_map) =
                    build_barrier_maps(non_term_ops, &egraph, &class_to_vreg);
                mark_branch_cond_barrier(
                    block.ops.last(),
                    non_term_count,
                    &egraph,
                    &class_to_vreg,
                    &mut arg_map,
                );
                let mut vreg_group =
                    assign_barrier_groups(&block_schedules[block_idx], &result_map, &arg_map);
                populate_effectful_operands(
                    &mut block_schedules[block_idx],
                    non_term_ops,
                    &egraph,
                    &class_to_vreg,
                    &mut vreg_group,
                    &mut next_vreg,
                );
            }
        }

        // Step 2: Compute global liveness.
        let global_liveness = crate::regalloc::global_liveness::compute_global_liveness(
            &block_schedules,
            &cfg_succs,
            &phi_uses,
        );

        // Step 3: Determine block params per block (excluded from reload insertion).
        let mut block_param_vregs_per_block =
            crate::regalloc::global_liveness::collect_block_param_vregs_per_block(
                func,
                &egraph,
                &class_to_vreg,
            );

        // Include fresh block param VRegs in the per-block sets.
        for (&(bid, _pidx), &fresh_vreg) in &block_param_vreg_overrides {
            let block_idx = func.blocks.iter().position(|b| b.id == bid).unwrap();
            block_param_vregs_per_block[block_idx].insert(fresh_vreg);
        }

        // Step 4: Assign cross-block spill slots.
        let spill_map = crate::regalloc::split::assign_cross_block_slots(
            &global_liveness,
            &block_schedules,
            &block_param_vregs_per_block,
        );
        let cross_block_slots = spill_map.num_slots;

        // Step 5: Build def_insts map for rematerialization.
        let def_insts: BTreeMap<VReg, ScheduledInst> = block_schedules
            .iter()
            .flatten()
            .map(|inst| (inst.dst, inst.clone()))
            .collect();

        // Step 6: Build vreg_classes map from all schedules.
        let vreg_classes =
            crate::regalloc::split::build_vreg_classes_from_schedules(&block_schedules);

        // Step 7: Compute block defs.
        let block_defs = crate::regalloc::split::compute_block_defs(&block_schedules);

        // Build function-level pre-colorings for filtering per block.
        let mut func_level_param_vregs = param_vregs.clone();
        {
            let all_scheduled: Vec<ScheduledInst> =
                block_schedules.iter().flatten().cloned().collect();
            add_shift_precolors(&all_scheduled, &mut func_level_param_vregs);
            add_div_precolors(&all_scheduled, &mut func_level_param_vregs);
        }

        // Step 8: Per-block allocation loop (RPO order).
        let mut shared_next_vreg = shared_next_vreg_start;
        let mut merged_vreg_to_reg: BTreeMap<VReg, Reg> = BTreeMap::new();
        let mut merged_callee_saved: Vec<Reg> = Vec::new();
        let mut spill_slot_counter = cross_block_slots;
        let mut block_rewritten_storage: Vec<Vec<ScheduledInst>> =
            vec![Vec::new(); func.blocks.len()];
        let mut block_rename_maps: Vec<BTreeMap<VReg, VReg>> =
            vec![BTreeMap::new(); func.blocks.len()];

        for &block_idx in &rpo_order {
            let block = &func.blocks[block_idx];

            // Step 8a: Insert cross-block spill/reload code.
            let (split_schedule, rename) = crate::regalloc::split::rewrite_block_for_splitting(
                &block_schedules[block_idx],
                block_idx,
                &global_liveness,
                &spill_map,
                &block_defs,
                &def_insts,
                &mut shared_next_vreg,
                &block_param_vregs_per_block[block_idx],
                &vreg_classes,
            );

            // Step 8a-ret: If this block ends with Ret, ensure the return value
            // is available in this block's schedule. The global liveness doesn't
            // include Ret operands (adding them breaks while loop liveness), so
            // rewrite_block_for_splitting won't insert a reload/remat for them.
            // Handle it here: if the ret value is from another block and not
            // already in the schedule, insert a remat (for Iconst/StackAddr) or
            // spill-load.
            let (split_schedule, mut rename) = {
                let mut schedule = split_schedule;
                let mut rename = rename;
                if let Some(EffectfulOp::Ret { val: Some(cid) }) = block.ops.last() {
                    let canon = egraph.unionfind.find_immutable(*cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        let in_schedule = schedule.iter().any(|i| i.dst == vreg)
                            || rename
                                .values()
                                .any(|&v| schedule.iter().any(|i| i.dst == v));
                        if !in_schedule {
                            // Value not in this block's schedule -- need remat or reload.
                            if let Some(def) = def_insts.get(&vreg) {
                                if crate::regalloc::spill::is_rematerializable(def) {
                                    let new_vreg = VReg(shared_next_vreg);
                                    shared_next_vreg += 1;
                                    schedule.push(ScheduledInst {
                                        op: def.op.clone(),
                                        dst: new_vreg,
                                        operands: def.operands.clone(),
                                    });
                                    rename.insert(vreg, new_vreg);
                                } else if let Some(&slot) = spill_map.vreg_to_slot.get(&vreg) {
                                    let new_vreg = VReg(shared_next_vreg);
                                    shared_next_vreg += 1;
                                    schedule.push(ScheduledInst {
                                        op: Op::SpillLoad(slot as i64),
                                        dst: new_vreg,
                                        operands: vec![],
                                    });
                                    rename.insert(vreg, new_vreg);
                                }
                            }
                        }
                    }
                }
                // Step 8a-phi: Ensure phi source VRegs (terminator args) are in the
                // schedule. The pass-through optimization in cross-block splitting
                // skips VRegs that are live-in/live-out but not used in the
                // schedule. Phi source VRegs need to be in registers for phi copy
                // emission, so insert remat/reload if missing.
                for &phi_vreg in &phi_uses[block_idx] {
                    let vreg = phi_vreg;
                    let in_schedule = schedule.iter().any(|i| i.dst == vreg)
                        || rename
                            .get(&vreg)
                            .is_some_and(|&v| schedule.iter().any(|i| i.dst == v));
                    if !in_schedule && let Some(def) = def_insts.get(&vreg) {
                        if crate::regalloc::spill::is_rematerializable(def) {
                            let new_vreg = VReg(shared_next_vreg);
                            shared_next_vreg += 1;
                            schedule.push(ScheduledInst {
                                op: def.op.clone(),
                                dst: new_vreg,
                                operands: def.operands.clone(),
                            });
                            rename.insert(vreg, new_vreg);
                        } else if let Some(&slot) = spill_map.vreg_to_slot.get(&vreg) {
                            let new_vreg = VReg(shared_next_vreg);
                            shared_next_vreg += 1;
                            schedule.push(ScheduledInst {
                                op: Op::SpillLoad(slot as i64),
                                dst: new_vreg,
                                operands: vec![],
                            });
                            rename.insert(vreg, new_vreg);
                        }
                    }
                }
                (schedule, rename)
            };

            // Step 8a-reorder: Move all BlockParam instructions to the front of
            // split_schedule. Block params receive their values from predecessor
            // phi copies before the block begins execution. Placing them at
            // position 0 ensures liveness analysis treats them as live from the
            // start of the block, preventing other instructions scheduled before
            // their original position from sharing the same physical register.
            let split_schedule = {
                let (mut bps, rest): (Vec<_>, Vec<_>) = split_schedule
                    .into_iter()
                    .partition(|inst| matches!(inst.op, Op::BlockParam(_, _, _)));
                bps.extend(rest);
                bps
            };

            // Step 8a-spill: Insert early spill/reload for distant barrier results.
            // Rebuild barrier maps with post-rename VRegs, then insert pairs.
            // Strip barrier pseudo-ops (StoreBarrier, VoidCallBarrier) and clear
            // operands from LoadResult/CallResult — they'll be re-populated with
            // post-rename VRegs after the early spill pass.
            let mut split_schedule: Vec<_> = split_schedule
                .into_iter()
                .filter(|inst| !matches!(inst.op, Op::StoreBarrier | Op::VoidCallBarrier))
                .map(|mut inst| {
                    if matches!(inst.op, Op::LoadResult(_, _) | Op::CallResult(_, _)) {
                        inst.operands.clear();
                    }
                    inst
                })
                .collect();
            {
                let non_term_count = if block.ops.is_empty() {
                    0
                } else {
                    block.ops.len() - 1
                };
                let non_term_ops = &block.ops[..non_term_count];
                if non_term_count > 0 {
                    // Build block_class_to_vreg with renames applied.
                    let mut bcv: BTreeMap<ClassId, VReg> = class_to_vreg
                        .iter()
                        .map(|(&cid, &vreg)| {
                            let renamed = rename.get(&vreg).copied().unwrap_or(vreg);
                            (cid, renamed)
                        })
                        .collect();
                    let (result_map, mut arg_map) = build_barrier_maps(non_term_ops, &egraph, &bcv);
                    mark_branch_cond_barrier(
                        block.ops.last(),
                        non_term_count,
                        &egraph,
                        &bcv,
                        &mut arg_map,
                    );
                    let mut vreg_group =
                        assign_barrier_groups(&split_schedule, &result_map, &arg_map);
                    insert_early_barrier_spills(
                        &mut split_schedule,
                        &result_map,
                        &arg_map,
                        &mut vreg_group,
                        &vreg_types,
                        &mut shared_next_vreg,
                        &mut spill_slot_counter,
                    );

                    // Re-sort by barrier group after inserting SpillStore/SpillLoad.
                    // SpillLoad must come before consumers in the same group (like
                    // LoadResult/CallResult), and SpillStore right after its source.
                    {
                        let mut indexed: Vec<(usize, ScheduledInst)> =
                            split_schedule.drain(..).enumerate().collect();
                        indexed.sort_by_key(|(orig_idx, inst)| {
                            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
                            let priority: u8 = match inst.op {
                                Op::Param(_, _) | Op::BlockParam(_, _, _) => 0,
                                Op::LoadResult(_, _) | Op::CallResult(_, _) | Op::SpillLoad(_) => 1,
                                _ => 2,
                            };
                            (g, priority, *orig_idx)
                        });
                        split_schedule.extend(indexed.into_iter().map(|(_, inst)| inst));
                    }

                    // Step 8a-effectful: Ensure effectful op operands are in
                    // the schedule. If an operand's VReg is not in this block's
                    // schedule (because global liveness didn't detect it as
                    // cross-block live), insert a remat instruction.
                    {
                        let sched_vregs: BTreeSet<VReg> =
                            split_schedule.iter().map(|i| i.dst).collect();
                        for op in non_term_ops {
                            let cids: Vec<ClassId> = match op {
                                EffectfulOp::Store { addr, val, .. } => vec![*addr, *val],
                                EffectfulOp::Load { addr, .. } => vec![*addr],
                                EffectfulOp::Call { args, .. } => args.clone(),
                                _ => continue,
                            };
                            for cid in cids {
                                let canon = egraph.unionfind.find_immutable(cid);
                                if let Some(&vreg) = bcv.get(&canon) {
                                    if sched_vregs.contains(&vreg) {
                                        continue;
                                    }
                                    // Try original (pre-rename) VReg for def lookup.
                                    let orig_vreg =
                                        class_to_vreg.get(&canon).copied().unwrap_or(vreg);
                                    if let Some(def) =
                                        def_insts.get(&vreg).or_else(|| def_insts.get(&orig_vreg))
                                        && crate::regalloc::spill::is_rematerializable(def)
                                    {
                                        let new_vreg = VReg(shared_next_vreg);
                                        shared_next_vreg += 1;
                                        split_schedule.push(ScheduledInst {
                                            op: def.op.clone(),
                                            dst: new_vreg,
                                            operands: def.operands.clone(),
                                        });
                                        bcv.insert(canon, new_vreg);
                                        rename.insert(orig_vreg, new_vreg);
                                    }
                                }
                            }
                        }
                    }

                    // Re-populate effectful-op operands with post-rename VRegs.
                    populate_effectful_operands(
                        &mut split_schedule,
                        non_term_ops,
                        &egraph,
                        &bcv,
                        &mut vreg_group,
                        &mut shared_next_vreg,
                    );

                    if crate::trace::is_enabled("sched") && crate::trace::fn_matches(&func.name) {
                        tracing::debug!(
                            target: "blitz::sched",
                            "[{}] block {block_idx} after markers:\n{}",
                            func.name,
                            crate::trace::format_schedule(&split_schedule, Some(&vreg_group)),
                        );
                    }
                }
            }

            // Step 8b: Per-block live_out for allocate():
            //   - phi source VRegs (from this block's terminator args): ensures the
            //     phi source values stay live at the terminator for phi copy emission.
            //   - block param VRegs for this block: forces them into the interference
            //     graph so they are not coalesced away, ensuring build_phi_copies can
            //     find their physical registers in merged_vreg_to_reg.
            //   - effectful op operands are NOT in live_out: they're operands on
            //     barrier instructions (LoadResult/CallResult/StoreBarrier/VoidCallBarrier).
            //   - Ret operands ARE in live_out (terminator, no barrier instruction).
            let mut block_live_out: BTreeSet<VReg> = phi_uses[block_idx]
                .iter()
                .map(|v| rename.get(v).copied().unwrap_or(*v))
                .chain(block_param_vregs_per_block[block_idx].iter().copied())
                .collect();
            if let Some(EffectfulOp::Ret { val: Some(cid) }) = block.ops.last() {
                let canon = egraph.unionfind.find_immutable(*cid);
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    let renamed = rename.get(&vreg).copied().unwrap_or(vreg);
                    block_live_out.insert(renamed);
                }
            }
            // Step 8c: Filter pre-colorings to VRegs in this block's schedule.
            let split_vreg_set: BTreeSet<VReg> = split_schedule
                .iter()
                .flat_map(|i| std::iter::once(i.dst).chain(i.operands.iter().copied()))
                .collect();

            let block_param_vregs: Vec<(VReg, Reg)> = func_level_param_vregs
                .iter()
                .filter(|&&(v, _)| split_vreg_set.contains(&v))
                .copied()
                .collect();

            // Step 8c2: Add call precolors specific to THIS block only.
            // Call-arg precolors must not leak from other blocks where the same
            // VReg may be used as a call arg in a different ABI position.
            let mut block_param_vregs = block_param_vregs;
            {
                let mut dummy_live_out: BTreeSet<VReg> = BTreeSet::new();
                add_call_precolors_for_block(
                    block,
                    &egraph,
                    &class_to_vreg,
                    &mut block_param_vregs,
                    &mut dummy_live_out,
                );
            }

            // Step 8d: Filter copy pairs to VRegs in this block.
            let block_copy_pairs: Vec<(VReg, VReg)> = copy_pairs
                .iter()
                .filter(|&&(a, b)| split_vreg_set.contains(&a) && split_vreg_set.contains(&b))
                .copied()
                .collect();

            // Step 8e: Per-block call points (local instruction indices).
            let block_call_points = collect_call_points_for_block(
                func,
                block_idx,
                &split_schedule,
                &class_to_vreg,
                &egraph,
            );
            let block_div_points = collect_div_clobber_points(&split_schedule);

            // Step 8f: Per-block loop depths.
            let block_loop_depths: BTreeMap<VReg, u32> = loop_depths
                .iter()
                .filter(|(v, _)| split_vreg_set.contains(v))
                .map(|(&v, &d)| (v, d))
                .collect();

            // Step 8g: Run per-block allocation.
            let block_result = allocate(
                &split_schedule,
                &block_param_vregs,
                &block_live_out,
                &block_copy_pairs,
                &block_loop_depths,
                &block_call_points,
                &block_div_points,
                opts.force_frame_pointer,
                &func.name,
            )
            .map_err(|e| CompileError {
                phase: "regalloc".into(),
                message: format!("block {}: {}", block.id, e),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: Some(block.id),
                    inst: None,
                }),
            })?;

            // Merge results: include VRegs that appear in either the
            // pre-allocation schedule OR the post-allocation instruction list.
            // Pre-alloc is needed because effectful op args reference VRegs
            // from the original schedule (even if the allocator rematerialized
            // them internally). Post-alloc is needed for new VRegs created by
            // spill/remat. The interference graph includes "phantom" entries
            // for all VReg indices up to the max; those get color 0 (RAX) and
            // must not be merged as they would overwrite correct assignments.
            let post_alloc_vregs: BTreeSet<VReg> = block_result
                .insts
                .iter()
                .flat_map(|i| std::iter::once(i.dst).chain(i.operands.iter().copied()))
                .collect();
            for (v, r) in &block_result.vreg_to_reg {
                if split_vreg_set.contains(v) || post_alloc_vregs.contains(v) {
                    merged_vreg_to_reg.insert(*v, *r);
                }
            }
            for &r in &block_result.callee_saved_used {
                if !merged_callee_saved.contains(&r) {
                    merged_callee_saved.push(r);
                }
            }

            // Rewrite with physical registers. Use the allocator's final
            // instruction list which includes intra-block spill/reload code.
            let rewritten_insts = rewrite_vregs(&block_result.insts, &block_result.vreg_to_reg);
            spill_slot_counter += block_result.spill_slots;
            block_rewritten_storage[block_idx] = rewritten_insts;
            block_rename_maps[block_idx] = rename;
        }

        merged_callee_saved.sort_by_key(|r| *r as u8);
        merged_callee_saved.dedup();

        let merged_result = RegAllocResult {
            vreg_to_reg: merged_vreg_to_reg,
            spill_slots: spill_slot_counter,
            callee_saved_used: merged_callee_saved,
            insts: vec![],               // per-block insts already consumed above
            unprecolored_params: vec![], // multi-block handles this separately
        };

        (merged_result, block_rewritten_storage, block_rename_maps)
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

        // The block that follows this one in emission order (for fallthrough).
        let next_block_id: Option<BlockId> = rpo_order
            .get(rpo_pos + 1)
            .map(|&next_idx| func.blocks[next_idx].id);

        // Build a block-local class_to_vreg that applies cross-block renames.
        // When a VReg was renamed (SpillLoad/remat) in this block, effectful ops
        // must use the renamed VReg to get the correct physical register.
        let block_class_to_vreg: BTreeMap<ClassId, VReg> = {
            let renames = &block_rename_maps[block_idx];
            let mut map: BTreeMap<ClassId, VReg> = if renames.is_empty() {
                class_to_vreg.clone()
            } else {
                class_to_vreg
                    .iter()
                    .map(|(&cid, &vreg)| {
                        let renamed = renames.get(&vreg).copied().unwrap_or(vreg);
                        (cid, renamed)
                    })
                    .collect()
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
                        map.insert(canon, fresh_vreg);
                    } else if let Some(&renamed) = renames.get(&fresh_vreg) {
                        // The override was reloaded into this block.
                        map.insert(canon, renamed);
                    }
                }
            }
            map
        };

        // Handle non-terminator effectful ops (loads, stores, calls).
        let non_term_count = if block.ops.is_empty() {
            0
        } else {
            block.ops.len() - 1
        };
        let non_term_ops = &block.ops[..non_term_count];

        // Build barrier maps using block_class_to_vreg (with cross-block renames
        // applied) so barrier operand caps match the renamed VRegs in the schedule.
        let (vreg_to_result_of_barrier, mut vreg_to_arg_of_barrier) =
            build_barrier_maps(non_term_ops, &egraph, &block_class_to_vreg);
        mark_branch_cond_barrier(
            block.ops.last(),
            non_term_count,
            &egraph,
            &block_class_to_vreg,
            &mut vreg_to_arg_of_barrier,
        );
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
        for inst in rewritten.iter() {
            if let Op::Param(param_idx, _) = &inst.op {
                if !param_vreg_set.contains(&inst.dst) {
                    let arg_locs = crate::x86::abi::assign_args(&func.param_types);
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
                &block_class_to_vreg,
                &regalloc_result,
                &extraction,
                func,
                &egraph.unionfind,
                rewritten,
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

        // Handle the terminator.
        let terminator = block.ops.last().expect("block must have terminator");
        let term_items = lower_terminator(
            terminator,
            next_block_id,
            &egraph,
            &class_to_vreg,
            &block_class_to_vreg,
            &block_param_map,
            &block_param_vreg_overrides,
            &regalloc_result,
            func,
            &mut next_label,
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
    let (flat_insts, is_short) =
        crate::emit::relax::relax_branches(&flat_insts, &label_positions, &inst_size);

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

// ── compile_to_ir_string() ───────────────────────────────────────────────────

/// Run phases 1-4b and return the IR as a human-readable string.
pub fn compile_to_ir_string(
    mut func: Function,
    opts: &CompileOptions,
) -> Result<String, CompileError> {
    use crate::ir::print::{PrintableBlock, PrintableGroup, print_function_ir};

    crate::trace::init_tracing();

    let mut egraph = func
        .egraph
        .take()
        .expect("Function must contain an EGraph; use FunctionBuilder::finalize()");
    let func = &func;

    // Phase 1: E-graph rewrite rules.
    let egraph_opts = EGraphOptions {
        phase1_limit: opts.phase1_limit,
        phase2_limit: opts.phase2_limit,
        phase3_limit: opts.phase3_limit,
        max_classes: 500_000,
    };
    crate::egraph::algebraic::propagate_block_params(func, &mut egraph);
    run_phases(&mut egraph, &egraph_opts).map_err(|e| CompileError {
        phase: "egraph".into(),
        message: e,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;
    // Second pass catches constants revealed by folding in the first pass,
    // then re-run algebraic rules to fold newly exposed constant expressions.
    crate::egraph::algebraic::propagate_block_params(func, &mut egraph);
    crate::egraph::algebraic::apply_algebraic_rules(&mut egraph);
    egraph.rebuild();

    // Build block param class map before extraction so params are roots.
    let block_param_map = build_block_param_class_map(&egraph);

    let mut all_roots = collect_roots(func);
    all_roots.extend(block_param_map.values().copied());
    all_roots.sort_by_key(|c| c.0);
    all_roots.dedup();

    // Phase 2: Extraction.
    let cost_model = CostModel::new(opts.opt_goal);
    let extraction = extract(&egraph, &all_roots, &cost_model).map_err(|e| CompileError {
        phase: "extraction".into(),
        message: e.to_string(),
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

    // Phase 3: Build per-block VRegInst lists.
    let mut class_to_vreg: BTreeMap<ClassId, VReg> = BTreeMap::new();
    let mut next_vreg: u32 = 0;
    let rpo_order = compute_rpo(func);

    let mut block_vreg_insts: Vec<Vec<crate::egraph::extract::VRegInst>> =
        vec![Vec::new(); func.blocks.len()];
    for &block_idx in &rpo_order {
        let block = &func.blocks[block_idx];
        let roots = collect_block_roots(block, &egraph);
        let block_id = block.id;
        let mut all_roots = roots;
        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block_id, pidx)) {
                all_roots.push(cid);
            }
        }
        all_roots.sort_by_key(|c| c.0);
        all_roots.dedup();
        let mut insts =
            vreg_insts_for_block(&extraction, &all_roots, &mut class_to_vreg, &mut next_vreg);

        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block_id, pidx)) {
                let canon = egraph.unionfind.find_immutable(cid);
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    if let Some(inst) = insts.iter_mut().find(|i| i.dst == vreg) {
                        inst.op = Op::BlockParam(
                            block_id,
                            pidx,
                            block.param_types[pidx as usize].clone(),
                        );
                        inst.operands.clear();
                    } else {
                        // VReg emitted by a prior block -- allocate a fresh one.
                        let fresh_vreg = VReg(next_vreg);
                        next_vreg += 1;
                        for inst in insts.iter_mut() {
                            for operand in inst.operands.iter_mut() {
                                if *operand == Some(vreg) {
                                    *operand = Some(fresh_vreg);
                                }
                            }
                        }
                        insts.push(crate::egraph::extract::VRegInst {
                            dst: fresh_vreg,
                            op: Op::BlockParam(
                                block_id,
                                pidx,
                                block.param_types[pidx as usize].clone(),
                            ),
                            operands: vec![],
                        });
                    }
                }
            }
        }

        block_vreg_insts[block_idx] = insts;
    }

    // Phase 4: Schedule per block.
    let mut block_schedules: Vec<Vec<ScheduledInst>> = vec![Vec::new(); func.blocks.len()];
    for (block_idx, insts) in block_vreg_insts.iter().enumerate() {
        let dag = ScheduleDag::build(insts);
        let sched = schedule(&dag);
        block_schedules[block_idx] = sched;
    }

    // Phase 4b: Reorder each block's schedule to respect effectful op barriers.
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let non_term_count = if block.ops.is_empty() {
            0
        } else {
            block.ops.len() - 1
        };
        if non_term_count == 0 {
            continue;
        }
        let non_term_ops = &block.ops[..non_term_count];

        let (vreg_to_result_of_barrier, mut vreg_to_arg_of_barrier) =
            build_barrier_maps(non_term_ops, &egraph, &class_to_vreg);
        mark_branch_cond_barrier(
            block.ops.last(),
            non_term_count,
            &egraph,
            &class_to_vreg,
            &mut vreg_to_arg_of_barrier,
        );

        let sched = &block_schedules[block_idx];
        let vreg_group =
            assign_barrier_groups(sched, &vreg_to_result_of_barrier, &vreg_to_arg_of_barrier);

        let mut indexed: Vec<(usize, &ScheduledInst)> = sched.iter().enumerate().collect();
        indexed.sort_by_key(|(orig_idx, inst)| {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            let param_order: u8 = match inst.op {
                Op::Param(_, _) => 0,
                Op::LoadResult(_, _) | Op::CallResult(_, _) => 1,
                _ => 2,
            };
            (g, param_order, *orig_idx)
        });
        let reordered: Vec<ScheduledInst> =
            indexed.into_iter().map(|(_, inst)| inst.clone()).collect();
        block_schedules[block_idx] = reordered;
    }

    // Build PrintableBlocks for the printer.
    let mut printable_blocks: Vec<PrintableBlock> = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let non_term_count = if block.ops.is_empty() {
            0
        } else {
            block.ops.len() - 1
        };
        let non_term_ops = &block.ops[..non_term_count];
        let num_barriers = non_term_ops.len();

        // Build barrier groups for this block's schedule.
        let (vreg_to_result_of_barrier, mut vreg_to_arg_of_barrier) =
            build_barrier_maps(non_term_ops, &egraph, &class_to_vreg);
        mark_branch_cond_barrier(
            block.ops.last(),
            non_term_count,
            &egraph,
            &class_to_vreg,
            &mut vreg_to_arg_of_barrier,
        );
        let vreg_group = assign_barrier_groups(
            &block_schedules[block_idx],
            &vreg_to_result_of_barrier,
            &vreg_to_arg_of_barrier,
        );

        // Partition scheduled insts into groups.
        let mut groups_insts: Vec<Vec<ScheduledInst>> = vec![Vec::new(); num_barriers + 1];
        for inst in &block_schedules[block_idx] {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            groups_insts[g].push(inst.clone());
        }

        // Build PrintableGroups: barrier k has pure ops from groups_insts[k] + barrier op k.
        let mut groups: Vec<PrintableGroup> = Vec::new();
        for k in 0..num_barriers {
            groups.push(PrintableGroup {
                pure_ops: groups_insts[k].clone(),
                barrier: Some(non_term_ops[k].clone()),
            });
        }
        // Trailing group (after last barrier).
        groups.push(PrintableGroup {
            pure_ops: groups_insts[num_barriers].clone(),
            barrier: None,
        });

        let terminator = block
            .ops
            .last()
            .expect("block must have terminator")
            .clone();

        printable_blocks.push(PrintableBlock {
            id: block.id,
            param_types: block.param_types.clone(),
            groups,
            terminator,
        });
    }

    Ok(print_function_ir(
        func,
        &printable_blocks,
        &class_to_vreg,
        &egraph.unionfind,
    ))
}

/// Compile multiple functions to IR strings.
pub fn compile_module_to_ir(
    mut functions: Vec<Function>,
    opts: &CompileOptions,
) -> Result<String, CompileError> {
    crate::inline::inline_module(&mut functions, opts);
    let mut results = Vec::new();
    for func in functions {
        results.push(compile_to_ir_string(func, opts)?);
    }
    Ok(results.join("\n"))
}

// ── compile_module() ──────────────────────────────────────────────────────────

/// Compile multiple functions into a single object file.
///
/// Each `Function` (with its embedded e-graph) is consumed and compiled independently.
pub fn compile_module(
    functions: Vec<Function>,
    opts: &CompileOptions,
) -> Result<ObjectFile, CompileError> {
    compile_module_with_globals(functions, opts, vec![], vec![])
}

/// Compile multiple functions into a single object file, with global variable definitions.
pub fn compile_module_with_globals(
    mut functions: Vec<Function>,
    opts: &CompileOptions,
    globals: Vec<crate::emit::object::GlobalInfo>,
    rodata: Vec<crate::emit::object::GlobalInfo>,
) -> Result<ObjectFile, CompileError> {
    crate::inline::inline_module(&mut functions, opts);

    // Collect global and rodata names so we can filter them from externals.
    let global_names: std::collections::HashSet<String> = globals
        .iter()
        .chain(rodata.iter())
        .map(|g| g.name.clone())
        .collect();

    let mut combined_code: Vec<u8> = Vec::new();
    let mut combined_relocs = Vec::new();
    let mut combined_funcs: Vec<FunctionInfo> = Vec::new();
    let mut combined_externals: Vec<String> = Vec::new();

    for func in functions {
        let obj = compile(func, opts, None)?;

        // Adjust relocation offsets by the current combined code offset.
        let base_offset = combined_code.len();
        for mut reloc in obj.relocations {
            reloc.offset += base_offset;
            combined_relocs.push(reloc);
        }

        // Adjust function offsets.
        for mut fi in obj.functions {
            fi.offset += base_offset;
            combined_funcs.push(fi);
        }

        combined_code.extend_from_slice(&obj.code);

        // Collect unique externals, excluding global variable names.
        for ext in obj.externals {
            if !combined_externals.contains(&ext) && !global_names.contains(&ext) {
                combined_externals.push(ext);
            }
        }
    }

    Ok(ObjectFile {
        code: combined_code,
        relocations: combined_relocs,
        functions: combined_funcs,
        externals: combined_externals,
        globals,
        rodata,
    })
}

/// A flat item emitted for a block: either a MachInst or a label binding.
pub(crate) enum BlockItem {
    Inst(MachInst),
    BindLabel(LabelId),
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
