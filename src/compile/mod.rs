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

use std::collections::{HashMap, HashSet};

use crate::egraph::cost::{CostModel, OptGoal};
use crate::egraph::extract::{VReg, VRegInst, extract, vreg_insts_for_block};
use crate::egraph::phases::{CompileOptions as EGraphOptions, run_phases};
use crate::emit::object::{FunctionInfo, ObjectFile};
use crate::emit::peephole::peephole;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::regalloc::allocator::{RegAllocResult, allocate};
use crate::regalloc::rewrite::rewrite_vregs;
use crate::schedule::scheduler::{ScheduleDag, ScheduledInst, schedule};
use crate::x86::abi::{compute_frame_layout, emit_epilogue, emit_prologue};
use crate::x86::encode::{Encoder, inst_size};
use crate::x86::inst::{LabelId, MachInst};
use crate::x86::reg::Reg;

mod cfg;
use cfg::{
    build_block_param_class_map, collect_block_roots, collect_externals, collect_phi_source_vregs,
    collect_roots, compute_copy_pairs, compute_loop_depths, compute_rpo,
};
mod effectful;
use effectful::lower_effectful_op;
mod lower;
use lower::lower_block_pure_ops;
mod precolor;
use precolor::{
    add_call_precolors, add_div_precolors, add_shift_precolors, assign_param_vregs_from_map,
    collect_call_points_for_block, collect_div_clobber_points,
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

// ── compile() ────────────────────────────────────────────────────────────────

/// Compile a single function to an object file.
///
/// Consumes the `Function` (including its embedded e-graph).
pub fn compile(
    mut func: Function,
    opts: &CompileOptions,
    mut sink: Option<&mut dyn DiagnosticSink>,
) -> Result<ObjectFile, CompileError> {
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
    run_phases(&mut egraph, &egraph_opts).map_err(|e| CompileError {
        phase: "egraph".into(),
        message: e,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

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

    // Collect all root ClassIds from all effectful ops across all blocks.
    let all_roots = collect_roots(func);

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
    let mut class_to_vreg: HashMap<ClassId, VReg> = HashMap::new();
    let mut next_vreg: u32 = 0;

    // Build the block param class map (needed for phi copy generation).
    let block_param_map = build_block_param_class_map(&egraph);

    // Compute RPO block ordering (indices into func.blocks).
    let rpo_order = compute_rpo(func);

    // Build per-block VRegInst lists in RPO order, stored by block index.
    let mut block_vreg_insts: Vec<Vec<VRegInst>> = vec![Vec::new(); func.blocks.len()];
    for &block_idx in &rpo_order {
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
        let insts =
            vreg_insts_for_block(&extraction, &all_roots, &mut class_to_vreg, &mut next_vreg);
        block_vreg_insts[block_idx] = insts;
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

    // Phase 5: Register allocation -- per-block with cross-block live range splitting.
    //
    // Single-block fast path: skip global liveness and run allocate() directly.
    // Multi-block path: compute global liveness, assign cross-block spill slots,
    // rewrite each block to insert spill/reload code at boundaries, then run
    // allocate() per block and merge results.
    let param_vregs = assign_param_vregs_from_map(func, &class_to_vreg, &egraph);

    // Build phi copy pairs from block parameter passing for coalescing.
    let copy_pairs = compute_copy_pairs(func, &class_to_vreg, &egraph, &block_param_map);

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
    let (regalloc_result, block_rewritten) = if func.blocks.len() == 1 {
        // --- Single-block fast path ---
        let all_scheduled: Vec<ScheduledInst> = block_schedules.iter().flatten().cloned().collect();

        let mut live_out: HashSet<VReg> = HashSet::new();
        collect_phi_source_vregs(func, &egraph, &class_to_vreg, &mut live_out);
        for &(vreg, _reg) in &param_vregs {
            live_out.insert(vreg);
        }

        let mut all_param_vregs = param_vregs.clone();
        add_shift_precolors(&all_scheduled, &mut all_param_vregs);
        add_div_precolors(&all_scheduled, &mut all_param_vregs);
        add_call_precolors(
            func,
            &egraph,
            &class_to_vreg,
            &mut all_param_vregs,
            &mut live_out,
        );

        let call_points =
            collect_call_points_for_block(func, 0, &all_scheduled, &class_to_vreg, &egraph);
        let div_points = collect_div_clobber_points(&all_scheduled);
        // Combine div clobber points with call clobber points for regalloc.
        let mut combined_points = call_points;
        combined_points.extend_from_slice(&div_points);

        let result = allocate(
            &all_scheduled,
            &all_param_vregs,
            &live_out,
            &copy_pairs,
            &loop_depths,
            &combined_points,
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

        let rewritten = vec![rewrite_vregs(&all_scheduled, &result.vreg_to_reg)];
        (result, rewritten)
    } else {
        // --- Multi-block path ---

        // Step 1: Compute CFG successors and phi uses per block.
        let cfg_succs = crate::regalloc::global_liveness::cfg_successors(func);
        let phi_uses = crate::regalloc::global_liveness::compute_phi_uses(
            func,
            &egraph.unionfind,
            &class_to_vreg,
        );

        // Step 2: Compute global liveness.
        let global_liveness = crate::regalloc::global_liveness::compute_global_liveness(
            &block_schedules,
            &cfg_succs,
            &phi_uses,
        );

        // Step 3: Determine block params per block (excluded from reload insertion).
        let block_param_vregs_per_block =
            crate::regalloc::global_liveness::collect_block_param_vregs_per_block(
                func,
                &egraph,
                &class_to_vreg,
            );

        // Step 4: Assign cross-block spill slots.
        let spill_map = crate::regalloc::split::assign_cross_block_slots(
            &global_liveness,
            &block_schedules,
            &block_param_vregs_per_block,
        );
        let cross_block_slots = spill_map.num_slots;

        // Step 5: Build def_insts map for rematerialization.
        let def_insts: HashMap<VReg, ScheduledInst> = block_schedules
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
        let mut dummy_live_out: HashSet<VReg> = HashSet::new();
        add_call_precolors(
            func,
            &egraph,
            &class_to_vreg,
            &mut func_level_param_vregs,
            &mut dummy_live_out,
        );

        // Step 8: Per-block allocation loop (RPO order).
        let mut shared_next_vreg = shared_next_vreg_start;
        let mut merged_vreg_to_reg: HashMap<VReg, Reg> = HashMap::new();
        let mut merged_callee_saved: Vec<Reg> = Vec::new();
        let mut spill_slot_counter = cross_block_slots;
        let mut block_rewritten_storage: Vec<Vec<ScheduledInst>> =
            vec![Vec::new(); func.blocks.len()];

        for &block_idx in &rpo_order {
            let block = &func.blocks[block_idx];

            // Step 8a: Insert cross-block spill/reload code.
            let (split_schedule, _rename) = crate::regalloc::split::rewrite_block_for_splitting(
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

            // Step 8b: Per-block live_out for allocate():
            //   - phi source VRegs (from this block's terminator args): ensures the
            //     phi source values stay live at the terminator for phi copy emission.
            //   - block param VRegs for this block: forces them into the interference
            //     graph so they are not coalesced away, ensuring build_phi_copies can
            //     find their physical registers in merged_vreg_to_reg.
            let block_live_out: HashSet<VReg> = phi_uses[block_idx]
                .iter()
                .chain(block_param_vregs_per_block[block_idx].iter())
                .copied()
                .collect();

            // Step 8c: Filter pre-colorings to VRegs in this block's schedule.
            let split_vreg_set: HashSet<VReg> = split_schedule
                .iter()
                .flat_map(|i| std::iter::once(i.dst).chain(i.operands.iter().copied()))
                .collect();

            let block_param_vregs: Vec<(VReg, Reg)> = func_level_param_vregs
                .iter()
                .filter(|&&(v, _)| split_vreg_set.contains(&v))
                .copied()
                .collect();

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
            // Combine call and div clobber points for regalloc.
            let mut block_combined_points = block_call_points;
            block_combined_points.extend_from_slice(&block_div_points);

            // Step 8f: Per-block loop depths.
            let block_loop_depths: HashMap<VReg, u32> = loop_depths
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
                &block_combined_points,
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

            // Merge results: only insert VRegs that actually appear in this
            // block's split schedule. The allocate() function builds an
            // interference graph with num_vregs = max_vreg_seen + 1, which
            // includes "phantom" entries for all VReg indices up to the max.
            // Those phantom entries get color 0 (RAX) and must not be merged
            // into merged_vreg_to_reg as they would overwrite correct
            // assignments from other blocks.
            for (v, r) in &block_result.vreg_to_reg {
                if split_vreg_set.contains(v) {
                    merged_vreg_to_reg.insert(*v, *r);
                }
            }
            for &r in &block_result.callee_saved_used {
                if !merged_callee_saved.contains(&r) {
                    merged_callee_saved.push(r);
                }
            }

            // Rewrite split_schedule with physical registers.
            // The split_schedule contains cross-block spill/reload instructions
            // with absolute slot numbers (0..cross_block_slots). Any intra-block
            // spill instructions inserted by allocate() internally use a separate
            // local copy and are not reflected here; their slot count is tracked
            // for frame layout purposes.
            let rewritten_insts = rewrite_vregs(&split_schedule, &block_result.vreg_to_reg);
            spill_slot_counter += block_result.spill_slots;
            block_rewritten_storage[block_idx] = rewritten_insts;
        }

        merged_callee_saved.sort_by_key(|r| *r as u8);
        merged_callee_saved.dedup();

        let merged_result = RegAllocResult {
            vreg_to_reg: merged_vreg_to_reg,
            spill_slots: spill_slot_counter,
            callee_saved_used: merged_callee_saved,
        };

        (merged_result, block_rewritten_storage)
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
    let param_vreg_set: HashSet<VReg> = param_vregs.iter().map(|(v, _)| *v).collect();

    // Compute frame layout early so spill lowering can use it during Phase 7.
    let frame_layout = compute_frame_layout(
        regalloc_result.spill_slots,
        &regalloc_result.callee_saved_used,
        0,
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
        let rewritten = &block_rewritten[block_idx];

        // The block that follows this one in emission order (for fallthrough).
        let next_block_id: Option<BlockId> = rpo_order
            .get(rpo_pos + 1)
            .map(|&next_idx| func.blocks[next_idx].id);

        // Handle non-terminator effectful ops (loads, stores, calls).
        let non_term_count = if block.ops.is_empty() {
            0
        } else {
            block.ops.len() - 1
        };
        let non_term_ops = &block.ops[..non_term_count];

        // Build maps for interleaving pure ops with calls:
        //   vreg_to_result_of_call[vreg] = K  means vreg is produced by call K
        //   vreg_to_arg_of_call[vreg]    = K  means vreg is an argument to call K
        //
        // Pure ops are placed in groups:
        //   group[0]   = before any call
        //   group[K+1] = after call K (because they use its result, or transitively)
        //
        // Additionally, an inst that computes an argument for call K is placed in
        // group[K] so it is emitted immediately before call K, preventing a later
        // call's arg-setup from clobbering the earlier call's arg register.
        let mut vreg_to_result_of_call: HashMap<VReg, usize> = HashMap::new();
        let mut vreg_to_arg_of_call: HashMap<VReg, usize> = HashMap::new();
        let mut call_op_indices: Vec<usize> = Vec::new();
        for (op_idx, op) in non_term_ops.iter().enumerate() {
            if let EffectfulOp::Call { args, results, .. } = op {
                let call_k = call_op_indices.len();
                call_op_indices.push(op_idx);
                for &result_cid in results {
                    let canon = egraph.unionfind.find_immutable(result_cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        vreg_to_result_of_call.insert(vreg, call_k);
                    }
                }
                for &arg_cid in args {
                    let canon = egraph.unionfind.find_immutable(arg_cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        // Only record the latest call that needs this vreg as arg,
                        // in case the same value is passed to multiple calls.
                        let entry = vreg_to_arg_of_call.entry(vreg).or_insert(call_k);
                        *entry = (*entry).max(call_k);
                    }
                }
            }
        }

        // Partition scheduled insts into groups relative to calls:
        //
        //   group[k]   = pure insts that are arguments to call k (emit just before call k)
        //   group[k+1] = pure insts that use results of call k (emit just after call k)
        //   group[0]   = all other pure insts (emit before all calls)
        //
        // Groups are 0..=num_calls. Group 0 is "before all calls / no dependency".
        // Group k (1..=num_calls) is "after call k-1 but before call k" (or after the last call).
        //
        // For argument insts: an inst that directly computes a call K argument is placed in group[K].
        // This ensures it's emitted right before call K, not before an earlier call clobbers the arg reg.
        //
        // For result-dependent insts: an inst that uses CallResult of call K goes in group[K+1].
        //
        // Transitive propagation: if inst A depends on inst B (via operand), A's group >= B's group.
        let num_calls = call_op_indices.len();
        // groups has num_calls+1 slots: 0..=num_calls.
        let mut groups: Vec<Vec<&ScheduledInst>> = vec![Vec::new(); num_calls + 1];
        if num_calls == 0 {
            // No calls: all pure insts go to group[0].
            for inst in rewritten.iter() {
                groups[0].push(inst);
            }
        } else {
            // Assign group for each instruction.
            //
            // Rules:
            //   1. If inst.dst is an arg to call K → min group K
            //   2. If inst uses a CallResult from call K → min group K+1
            //   3. Transitive forward: if operand's def is in group G → min group G
            //   4. Transitive backward: if inst is in group G and its dep is in group < G →
            //      the dep must also be in group G (so it's recomputed after the same call)
            //
            // Rules 1-3 are computed in a forward pass. Rule 4 requires propagation.
            // We iterate until no more changes occur (fixpoint).

            // Forward pass: compute initial groups.
            let mut vreg_group: HashMap<VReg, usize> = HashMap::new();
            for inst in rewritten.iter() {
                let mut min_group: usize = 0;
                for &operand_vreg in &inst.operands {
                    if let Some(&call_k) = vreg_to_result_of_call.get(&operand_vreg) {
                        min_group = min_group.max(call_k + 1);
                    }
                    if let Some(&og) = vreg_group.get(&operand_vreg) {
                        min_group = min_group.max(og);
                    }
                }
                if min_group == 0 {
                    if let Some(&arg_call_k) = vreg_to_arg_of_call.get(&inst.dst) {
                        min_group = arg_call_k;
                    }
                }
                vreg_group.insert(inst.dst, min_group);
            }

            // Backward propagation: if inst W is in group G (G > 0) and its dep inst D is
            // in group < G, then D must also be in group G (so D is recomputed after the
            // same call as W's group). This ensures that dependencies of "late" insts are
            // not left in caller-saved registers that would be clobbered.
            //
            // We propagate backward: for each inst in decreasing group order, pull all
            // its pure-op dependencies up to the same group. Iterate until fixpoint.
            let mut changed = true;
            while changed {
                changed = false;
                for inst in rewritten.iter() {
                    let current_group = *vreg_group.get(&inst.dst).unwrap_or(&0);
                    if current_group == 0 {
                        continue; // Already in lowest group; no backward pull.
                    }
                    for &operand_vreg in &inst.operands {
                        if vreg_to_result_of_call.contains_key(&operand_vreg) {
                            // This operand IS a call result; skip backward propagation.
                            continue;
                        }
                        if let Some(dep_group) = vreg_group.get_mut(&operand_vreg) {
                            if *dep_group < current_group {
                                *dep_group = current_group;
                                changed = true;
                            }
                        }
                    }
                }
            }

            // Assign groups to actual group vectors.
            for inst in rewritten.iter() {
                let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
                groups[g].push(inst);
            }
        }

        // Emit pure ops and calls in interleaved order.
        //
        // Emission order for each call K (0-based):
        //   1. group[K] pure ops (arguments to call K and insts between call K-1 and K)
        //   2. call K (effectful)
        //   3. (group[K+1] is emitted as step 1 of the next iteration)
        //
        // group[0] is emitted as the "before first call" group.
        // group[num_calls] (insts after the last call) is emitted after the last call.
        let mut all_insts: Vec<MachInst> = Vec::new();

        let lower_group = |group: &[&ScheduledInst],
                           regalloc_result: &RegAllocResult,
                           func: &Function,
                           param_vreg_set: &HashSet<VReg>,
                           frame_layout: &crate::x86::abi::FrameLayout|
         -> Result<Vec<MachInst>, CompileError> {
            lower_block_pure_ops(
                &group.iter().map(|&i| i.clone()).collect::<Vec<_>>(),
                regalloc_result,
                func,
                param_vreg_set,
                frame_layout,
            )
        };

        // When there are no calls, emit all pure ops (group[0]) before any
        // loads/stores, so that registers used as address indices are
        // materialized before the instructions that read them.
        let mut call_k = 0usize;
        if num_calls == 0 {
            let pre_insts = lower_group(
                &groups[0],
                &regalloc_result,
                func,
                &param_vreg_set,
                &frame_layout,
            )?;
            all_insts.extend(pre_insts);
        }

        for (op_idx, op) in non_term_ops.iter().enumerate() {
            if call_op_indices.get(call_k) == Some(&op_idx) {
                // Emit group[call_k]: pure ops that must come right before call k.
                // (For call_k=0, this is group[0] = pre-call ops and arg setups.)
                let pre_insts = lower_group(
                    &groups[call_k],
                    &regalloc_result,
                    func,
                    &param_vreg_set,
                    &frame_layout,
                )?;
                all_insts.extend(pre_insts);
                // Emit call k.
                let extra =
                    lower_effectful_op(op, &class_to_vreg, &regalloc_result, &extraction, func)?;
                all_insts.extend(extra);
                call_k += 1;
            } else {
                // Non-call effectful op (load/store): emit inline.
                // Pure ops are already emitted before this loop (no-call case)
                // or interleaved with calls (call case).
                let extra =
                    lower_effectful_op(op, &class_to_vreg, &regalloc_result, &extraction, func)?;
                all_insts.extend(extra);
            }
        }
        // Emit the last group (pure ops that depend on the last call's results).
        // In the no-call case, group[0] was already emitted above, so this is a no-op.
        if num_calls > 0 {
            let post_insts = lower_group(
                &groups[call_k],
                &regalloc_result,
                func,
                &param_vreg_set,
                &frame_layout,
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
            &block_param_map,
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
    let mut label_positions: HashMap<LabelId, usize> = HashMap::new();

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
    })
}

// ── compile_module() ──────────────────────────────────────────────────────────

/// Compile multiple functions into a single object file.
///
/// Each `Function` (with its embedded e-graph) is consumed and compiled independently.
pub fn compile_module(
    functions: Vec<Function>,
    opts: &CompileOptions,
) -> Result<ObjectFile, CompileError> {
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

        // Collect unique externals.
        for ext in obj.externals {
            if !combined_externals.contains(&ext) {
                combined_externals.push(ext);
            }
        }
    }

    Ok(ObjectFile {
        code: combined_code,
        relocations: combined_relocs,
        functions: combined_funcs,
        externals: combined_externals,
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
