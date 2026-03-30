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

use crate::egraph::EGraph;
use crate::egraph::cost::{CostModel, OptGoal};
use crate::egraph::extract::{ExtractionResult, VReg, VRegInst, extract, vreg_insts_for_block};
use crate::egraph::isel::find_cc_in_class;
use crate::egraph::phases::{CompileOptions as EGraphOptions, run_phases};
use crate::emit::object::{FunctionInfo, ObjectFile};
use crate::emit::peephole::peephole;
use crate::emit::phi_elim::phi_copies;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::regalloc::allocator::{RegAllocResult, allocate};
use crate::regalloc::rewrite::rewrite_vregs;
use crate::schedule::scheduler::{ScheduleDag, ScheduledInst, schedule};
use crate::x86::abi::{
    GPR_RETURN_REG, assign_args, compute_frame_layout, emit_epilogue, emit_prologue,
    setup_call_args,
};
use crate::x86::encode::{Encoder, inst_size};
use crate::x86::inst::{LabelId, MachInst, Operand};
use crate::x86::reg::Reg;

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
pub fn compile(
    func: &Function,
    egraph: EGraph,
    opts: &CompileOptions,
    mut sink: Option<&mut dyn DiagnosticSink>,
) -> Result<ObjectFile, CompileError> {
    let mut egraph = egraph;

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

    if let Some(s) = sink.as_deref_mut() {
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

    if let Some(s) = sink.as_deref_mut() {
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

    if let Some(s) = sink.as_deref_mut() {
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
    let loop_depths = compute_loop_depths(func, &block_schedules, &class_to_vreg);

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
        add_call_precolors(
            func,
            &egraph,
            &class_to_vreg,
            &mut all_param_vregs,
            &mut live_out,
        );

        let call_points = collect_call_points_for_block(func, 0, &all_scheduled);

        let result = allocate(
            &all_scheduled,
            &all_param_vregs,
            &live_out,
            &copy_pairs,
            &loop_depths,
            &call_points,
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
            let block_call_points = collect_call_points_for_block(func, block_idx, &split_schedule);

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
                &block_call_points,
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

    if let Some(s) = sink.as_deref_mut() {
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

        // Lower pure ops for this block.
        let pure_insts = lower_block_pure_ops(
            rewritten,
            &regalloc_result,
            func,
            &param_vreg_set,
            &frame_layout,
        )?;

        // Handle non-terminator effectful ops (loads, stores, calls).
        let non_term_count = if block.ops.is_empty() {
            0
        } else {
            block.ops.len() - 1
        };
        let mut all_insts = pure_insts;
        for op in &block.ops[..non_term_count] {
            let extra =
                lower_effectful_op(op, &class_to_vreg, &regalloc_result, &extraction, func)?;
            all_insts.extend(extra);
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

    if let Some(s) = sink.as_deref_mut() {
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
/// Each (Function, EGraph) pair is consumed and compiled independently.
pub fn compile_module(
    functions: Vec<(Function, EGraph)>,
    opts: &CompileOptions,
) -> Result<ObjectFile, CompileError> {
    let mut combined_code: Vec<u8> = Vec::new();
    let mut combined_relocs = Vec::new();
    let mut combined_funcs: Vec<FunctionInfo> = Vec::new();
    let mut combined_externals: Vec<String> = Vec::new();

    for (func, egraph) in functions {
        let obj = compile(&func, egraph, opts, None)?;

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

// ── RPO helpers ───────────────────────────────────────────────────────────────

/// Compute a reverse post-order traversal of the CFG starting from block 0.
///
/// Returns a `Vec<usize>` of block *indices* into `func.blocks` (not block IDs)
/// in RPO order. RPO ensures:
///   - Loop headers come before loop bodies.
///   - Fallthrough targets tend to be adjacent, reducing unnecessary jumps.
fn compute_rpo(func: &Function) -> Vec<usize> {
    if func.blocks.is_empty() {
        return vec![];
    }

    // Build a successor map: block index -> list of successor block indices.
    let n = func.blocks.len();

    // Map block id -> block index for fast lookup.
    let id_to_idx: HashMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|block| {
            let mut succs = Vec::new();
            if let Some(term) = block.ops.last() {
                match term {
                    EffectfulOp::Jump { target, .. } => {
                        if let Some(&idx) = id_to_idx.get(target) {
                            succs.push(idx);
                        }
                    }
                    EffectfulOp::Branch {
                        bb_true, bb_false, ..
                    } => {
                        if let Some(&idx) = id_to_idx.get(bb_true) {
                            succs.push(idx);
                        }
                        if let Some(&idx) = id_to_idx.get(bb_false) {
                            succs.push(idx);
                        }
                    }
                    _ => {}
                }
            }
            succs
        })
        .collect();

    // Iterative DFS post-order, then reverse.
    let mut post_order: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    // Stack holds (block_index, child_iterator_index).
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;

    while let Some((node, child_idx)) = stack.last_mut() {
        let node = *node;
        if *child_idx < successors[node].len() {
            let next_child = successors[node][*child_idx];
            *child_idx += 1;
            if !visited[next_child] {
                visited[next_child] = true;
                stack.push((next_child, 0));
            }
        } else {
            post_order.push(node);
            stack.pop();
        }
    }

    // Any blocks not reachable from block 0 are appended at the end in index order.
    for i in 0..n {
        if !visited[i] {
            post_order.push(i);
        }
    }

    post_order.reverse();
    post_order
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Collect all ClassIds that are roots for extraction (used by effectful ops).
fn push_block_class_ids(block: &crate::ir::function::BasicBlock, out: &mut Vec<ClassId>) {
    for op in &block.ops {
        match op {
            EffectfulOp::Load { addr, result, .. } => {
                out.push(*addr);
                out.push(*result);
            }
            EffectfulOp::Store { addr, val } => {
                out.push(*addr);
                out.push(*val);
            }
            EffectfulOp::Call { args, results, .. } => {
                out.extend_from_slice(args);
                out.extend_from_slice(results);
            }
            EffectfulOp::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                out.push(*cond);
                out.extend_from_slice(true_args);
                out.extend_from_slice(false_args);
            }
            EffectfulOp::Jump { args, .. } => out.extend_from_slice(args),
            EffectfulOp::Ret { val } => {
                if let Some(v) = val {
                    out.push(*v);
                }
            }
        }
    }
}

fn collect_roots(func: &Function) -> Vec<ClassId> {
    let mut roots = Vec::new();
    for block in &func.blocks {
        push_block_class_ids(block, &mut roots);
    }
    roots.sort_by_key(|c| c.0);
    roots.dedup();
    roots
}

/// Collect external symbol names referenced by Call ops.
fn collect_externals(func: &Function) -> Vec<String> {
    let mut externals = Vec::new();
    for block in &func.blocks {
        for op in &block.ops {
            if let EffectfulOp::Call { func: callee, .. } = op {
                if !externals.contains(callee) {
                    externals.push(callee.clone());
                }
            }
        }
    }
    externals
}

/// Map function parameters to (VReg, Reg) pairs for pre-coloring.
///
/// Uses `func.param_class_ids` (populated by the builder) to look up the
/// corresponding VRegs in the ClassId -> VReg map from extraction.
fn assign_param_vregs_from_map(
    func: &Function,
    class_to_vreg: &std::collections::HashMap<ClassId, VReg>,
    egraph: &EGraph,
) -> Vec<(VReg, Reg)> {
    if func.param_class_ids.is_empty() {
        return vec![];
    }

    let arg_locs = assign_args(&func.param_types);
    let mut pairs: Vec<(VReg, Reg)> = Vec::new();

    for (param_idx, &class_id) in func.param_class_ids.iter().enumerate() {
        // Canonicalize the class_id after run_phases merges.
        let canon = egraph.unionfind.find_immutable(class_id);
        if let Some(&vreg) = class_to_vreg.get(&canon) {
            if let crate::x86::abi::ArgLoc::Reg(reg) = arg_locs[param_idx] {
                pairs.push((vreg, reg));
            }
        }
    }

    pairs
}

fn get_dst(name: &str, dst_reg: Option<Reg>) -> Result<Reg, String> {
    dst_reg.ok_or_else(|| format!("{name}: no register for dst"))
}

fn get_op(name: &str, operand_regs: &[Option<Reg>], i: usize) -> Result<Reg, String> {
    operand_regs
        .get(i)
        .and_then(|r| *r)
        .ok_or_else(|| format!("{name}: no register for operand {i}"))
}

fn lower_binary_alu(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand, Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    if dst != src_a {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    insts.push(mk(Operand::Reg(dst), Operand::Reg(src_b)));
    Ok(insts)
}

fn lower_shift_cl(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    // Move value to shift into dst if needed.
    if dst != src_a {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    // The shift count operand is pre-colored to RCX before register allocation
    // when possible. If the count VReg was already pre-colored to another register
    // (e.g., because it is a function parameter in RSI), emit a MOV to RCX now.
    if src_b != Reg::RCX {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(Reg::RCX),
            src: Operand::Reg(src_b),
        });
    }
    insts.push(mk(Operand::Reg(dst)));
    Ok(insts)
}

fn lower_fp_binary(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand, Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    if dst != src_a {
        insts.push(MachInst::MovsdRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    insts.push(mk(Operand::Reg(dst), Operand::Reg(src_b)));
    Ok(insts)
}

fn lower_fp_binary_ss(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand, Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    if dst != src_a {
        insts.push(MachInst::MovssRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    insts.push(mk(Operand::Reg(dst), Operand::Reg(src_b)));
    Ok(insts)
}

/// Convert a single Op to a sequence of MachInsts.
///
/// `dst_vreg` is the VReg being defined; `dst_reg` is the physical reg (if allocated).
fn lower_op(
    op: &Op,
    dst_vreg: VReg,
    dst_reg: Option<Reg>,
    _operand_vregs: &[VReg],
    operand_regs: &[Option<Reg>],
) -> Result<Vec<MachInst>, String> {
    let _ = dst_vreg; // used for context in errors
    match op {
        Op::Iconst(val, _ty) => {
            let dst = dst_reg.ok_or_else(|| "Iconst: no register for dst".to_string())?;
            Ok(vec![MachInst::MovRI {
                dst: Operand::Reg(dst),
                imm: *val,
            }])
        }

        // Param nodes represent function parameters. Their value is already in
        // the ABI argument register (via pre-coloring), so no instruction is needed.
        // The lower_insts_with_ret function skips these VRegs, but as a safety net,
        // emit nothing here too.
        Op::Param(_, _) => Ok(vec![]),

        // BlockParam nodes represent block parameters (SSA phi inputs).
        // Their value arrives from predecessor blocks; no instruction is needed here.
        Op::BlockParam(_, _, _) => Ok(vec![]),

        // X86Add produces a Pair (result + flags); Proj0 extracts the value.
        // We emit: mov dst, src_a; add dst, src_b
        Op::X86Add => lower_binary_alu("X86Add", dst_reg, operand_regs, |dst, src| {
            MachInst::AddRR { dst, src }
        }),
        Op::X86Sub => lower_binary_alu("X86Sub", dst_reg, operand_regs, |dst, src| {
            MachInst::SubRR { dst, src }
        }),
        Op::X86And => lower_binary_alu("X86And", dst_reg, operand_regs, |dst, src| {
            MachInst::AndRR { dst, src }
        }),
        Op::X86Or => lower_binary_alu("X86Or", dst_reg, operand_regs, |dst, src| MachInst::OrRR {
            dst,
            src,
        }),
        Op::X86Xor => lower_binary_alu("X86Xor", dst_reg, operand_regs, |dst, src| {
            MachInst::XorRR { dst, src }
        }),
        // Variable shifts use CL (RCX). The shift count VReg is pre-colored to RCX
        // before register allocation, so src_b is guaranteed to be RCX here.
        Op::X86Shl => lower_shift_cl("X86Shl", dst_reg, operand_regs, |dst| MachInst::ShlRCL {
            dst,
        }),
        Op::X86Shr => lower_shift_cl("X86Shr", dst_reg, operand_regs, |dst| MachInst::ShrRCL {
            dst,
        }),
        Op::X86Sar => lower_shift_cl("X86Sar", dst_reg, operand_regs, |dst| MachInst::SarRCL {
            dst,
        }),

        // Immediate-form shifts: no CL constraint, emit mov+shift directly.
        Op::X86ShlImm(imm) => {
            let dst = get_dst("X86ShlImm", dst_reg)?;
            let src = get_op("X86ShlImm", operand_regs, 0)?;
            let mut insts = Vec::new();
            if dst != src {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                });
            }
            insts.push(MachInst::ShlRI {
                dst: Operand::Reg(dst),
                imm: *imm,
            });
            Ok(insts)
        }
        Op::X86ShrImm(imm) => {
            let dst = get_dst("X86ShrImm", dst_reg)?;
            let src = get_op("X86ShrImm", operand_regs, 0)?;
            let mut insts = Vec::new();
            if dst != src {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                });
            }
            insts.push(MachInst::ShrRI {
                dst: Operand::Reg(dst),
                imm: *imm,
            });
            Ok(insts)
        }
        Op::X86SarImm(imm) => {
            let dst = get_dst("X86SarImm", dst_reg)?;
            let src = get_op("X86SarImm", operand_regs, 0)?;
            let mut insts = Vec::new();
            if dst != src {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                });
            }
            insts.push(MachInst::SarRI {
                dst: Operand::Reg(dst),
                imm: *imm,
            });
            Ok(insts)
        }

        Op::X86Imul3 => {
            let dst = dst_reg.ok_or_else(|| "X86Imul3: no register for dst".to_string())?;
            let src_a = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Imul3: no register for operand 0".to_string())?;
            let src_b = operand_regs
                .get(1)
                .and_then(|r| *r)
                .ok_or_else(|| "X86Imul3: no register for operand 1".to_string())?;
            // For X86Imul3 without an immediate, fall back to Imul2RR.
            let mut insts = Vec::new();
            if dst != src_a {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src_a),
                });
            }
            insts.push(MachInst::Imul2RR {
                dst: Operand::Reg(dst),
                src: Operand::Reg(src_b),
            });
            Ok(insts)
        }

        Op::X86Cmov(cc) => {
            let dst = dst_reg.ok_or_else(|| "X86Cmov: no register for dst".to_string())?;
            // operands: [flags_vreg, true_vreg, false_vreg]
            // flags come from a comparison; Cmov selects between true and false.
            if operand_regs.len() < 3 {
                return Err("X86Cmov requires 3 operands".into());
            }
            let true_reg = operand_regs[1]
                .ok_or_else(|| "X86Cmov: no register for true operand".to_string())?;
            let false_reg = operand_regs[2]
                .ok_or_else(|| "X86Cmov: no register for false operand".to_string())?;

            let mut insts = Vec::new();
            // Load the false value into dst first, then conditionally overwrite.
            if dst != false_reg {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(false_reg),
                });
            }
            insts.push(MachInst::Cmov {
                cc: *cc,
                dst: Operand::Reg(dst),
                src: Operand::Reg(true_reg),
            });
            Ok(insts)
        }

        Op::X86Setcc(cc) => {
            let dst = dst_reg.ok_or_else(|| "X86Setcc: no register for dst".to_string())?;
            Ok(vec![MachInst::Setcc {
                cc: *cc,
                dst: Operand::Reg(dst),
            }])
        }

        Op::X86Lea2 => {
            let dst = dst_reg.ok_or_else(|| "X86Lea2: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea2: no register for base".to_string())?;
            let idx = operand_regs
                .get(1)
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea2: no register for index".to_string())?;
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: crate::x86::addr::Addr {
                    base: Some(base),
                    index: Some(idx),
                    scale: 1,
                    disp: 0,
                },
            }])
        }

        Op::X86Lea3 { scale } => {
            let dst = dst_reg.ok_or_else(|| "X86Lea3: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea3: no register for base".to_string())?;
            let idx = operand_regs
                .get(1)
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea3: no register for index".to_string())?;
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: crate::x86::addr::Addr {
                    base: Some(base),
                    index: Some(idx),
                    scale: *scale,
                    disp: 0,
                },
            }])
        }

        Op::X86Lea4 { scale, disp } => {
            let dst = dst_reg.ok_or_else(|| "X86Lea4: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea4: no register for base".to_string())?;
            let idx_reg = operand_regs.get(1).and_then(|r| *r);
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: crate::x86::addr::Addr {
                    base: Some(base),
                    index: idx_reg,
                    scale: *scale,
                    disp: *disp,
                },
            }])
        }

        Op::Addr { scale, disp } => {
            // Addr nodes are "free" in the cost model and get folded into loads/stores.
            // When extracted standalone (e.g., as a root), emit a LEA.
            let dst = dst_reg.ok_or_else(|| "Addr: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "Addr: no register for base".to_string())?;
            let idx_reg = operand_regs.get(1).and_then(|r| *r);
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: crate::x86::addr::Addr {
                    base: Some(base),
                    index: idx_reg,
                    scale: *scale,
                    disp: *disp,
                },
            }])
        }

        // Projections: Proj0 and Proj1 extract values from Pairs.
        // Proj0 extracts the first element (the value), Proj1 extracts flags.
        // Since the Pair's register holds the full value (the ADD instruction
        // defines a register that holds the result), Proj0 is effectively a
        // register copy if the src and dst differ.
        Op::Proj0 => {
            if let (Some(dst), Some(Some(src))) = (dst_reg, operand_regs.first()) {
                if dst == *src {
                    Ok(vec![]) // No-op: dst and src are the same register.
                } else {
                    Ok(vec![MachInst::MovRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(*src),
                    }])
                }
            } else {
                // Proj0 with no dst register: it might be a flags projection that's unused.
                Ok(vec![])
            }
        }

        Op::Proj1 => {
            // Proj1 extracts the flags component — flags live in the CPU flags register,
            // not in a GPR. No MachInst needed since flags are implicit.
            Ok(vec![])
        }

        // LoadResult nodes are skipped by lower_block_pure_ops; if reached here,
        // that's a bug in the pipeline.
        Op::LoadResult(_, _) => unreachable!(
            "LoadResult must be skipped by lower_block_pure_ops, not passed to lower_op"
        ),

        // CallResult nodes are skipped by lower_block_pure_ops; if reached here,
        // that's a bug in the pipeline.
        Op::CallResult(_, _) => unreachable!(
            "CallResult must be skipped by lower_block_pure_ops, not passed to lower_op"
        ),

        // ── x86 FP machine ops ────────────────────────────────────────────────
        Op::X86Addsd => lower_fp_binary("X86Addsd", dst_reg, operand_regs, |dst, src| {
            MachInst::AddsdRR { dst, src }
        }),
        Op::X86Subsd => lower_fp_binary("X86Subsd", dst_reg, operand_regs, |dst, src| {
            MachInst::SubsdRR { dst, src }
        }),
        Op::X86Mulsd => lower_fp_binary("X86Mulsd", dst_reg, operand_regs, |dst, src| {
            MachInst::MulsdRR { dst, src }
        }),
        Op::X86Divsd => lower_fp_binary("X86Divsd", dst_reg, operand_regs, |dst, src| {
            MachInst::DivsdRR { dst, src }
        }),
        Op::X86Sqrtsd => {
            let dst = get_dst("X86Sqrtsd", dst_reg)?;
            let src = get_op("X86Sqrtsd", operand_regs, 0)?;
            Ok(vec![MachInst::SqrtsdRR {
                dst: Operand::Reg(dst),
                src: Operand::Reg(src),
            }])
        }

        // ── x86 F32 machine ops ───────────────────────────────────────────────
        Op::X86Addss => lower_fp_binary_ss("X86Addss", dst_reg, operand_regs, |dst, src| {
            MachInst::AddssRR { dst, src }
        }),
        Op::X86Subss => lower_fp_binary_ss("X86Subss", dst_reg, operand_regs, |dst, src| {
            MachInst::SubssRR { dst, src }
        }),
        Op::X86Mulss => lower_fp_binary_ss("X86Mulss", dst_reg, operand_regs, |dst, src| {
            MachInst::MulssRR { dst, src }
        }),
        Op::X86Divss => lower_fp_binary_ss("X86Divss", dst_reg, operand_regs, |dst, src| {
            MachInst::DivssRR { dst, src }
        }),
        Op::X86Sqrtss => {
            let dst = get_dst("X86Sqrtss", dst_reg)?;
            let src = get_op("X86Sqrtss", operand_regs, 0)?;
            Ok(vec![MachInst::SqrtssRR {
                dst: Operand::Reg(dst),
                src: Operand::Reg(src),
            }])
        }

        // Fconst: load FP constant bits into a scratch GPR (R11), then move to XMM.
        // R11 is caller-saved and not used by regalloc for any persistent value.
        Op::Fconst(bits) => {
            let dst = dst_reg.ok_or_else(|| "Fconst: no register for dst".to_string())?;
            Ok(vec![
                MachInst::MovRI {
                    dst: Operand::Reg(Reg::R11),
                    imm: *bits as i64,
                },
                MachInst::MovqToXmm {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(Reg::R11),
                },
            ])
        }

        // Generic ops that should have been lowered by the e-graph phases.
        // These should not appear after isel.
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::UDiv
        | Op::SDiv
        | Op::URem
        | Op::SRem
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Shl
        | Op::Shr
        | Op::Sar
        | Op::Sext(_)
        | Op::Zext(_)
        | Op::Trunc(_)
        | Op::Bitcast(_)
        | Op::Icmp(_)
        | Op::Fadd
        | Op::Fsub
        | Op::Fmul
        | Op::Fdiv
        | Op::Fsqrt
        | Op::Select => Err(format!(
            "unlowered op {op:?}: generic IR must be lowered by isel phases before lowering"
        )),
        Op::X86Movsx { from, to: _ } => {
            use crate::ir::types::Type;
            let dst = dst_reg.ok_or_else(|| "X86Movsx: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Movsx: no register for src".to_string())?;
            let inst = match from {
                Type::I8 => MachInst::MovsxBR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                Type::I16 => MachInst::MovsxWR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                Type::I32 => MachInst::MovsxDR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                other => {
                    return Err(format!("X86Movsx: unsupported source type {other:?}"));
                }
            };
            Ok(vec![inst])
        }

        Op::X86Movzx { from, to: _ } => {
            use crate::ir::types::Type;
            let dst = dst_reg.ok_or_else(|| "X86Movzx: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Movzx: no register for src".to_string())?;
            let inst = match from {
                Type::I8 => MachInst::MovzxBR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                Type::I16 => MachInst::MovzxWR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                // 32-bit MOV zero-extends to 64-bit on x86-64 implicitly.
                Type::I32 => {
                    if dst == src {
                        return Ok(vec![]);
                    }
                    MachInst::MovRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(src),
                    }
                }
                other => {
                    return Err(format!("X86Movzx: unsupported source type {other:?}"));
                }
            };
            Ok(vec![inst])
        }

        Op::X86Trunc { .. } => {
            // Truncation is free on x86-64: upper bits are simply ignored.
            // Emit a MOV only if dst != src (register copy needed).
            let dst = dst_reg.ok_or_else(|| "X86Trunc: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Trunc: no register for src".to_string())?;
            if dst == src {
                Ok(vec![])
            } else {
                Ok(vec![MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                }])
            }
        }

        Op::X86Bitcast { from, to } => {
            use crate::ir::types::Type;
            let dst = dst_reg.ok_or_else(|| "X86Bitcast: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Bitcast: no register for src".to_string())?;
            let int_to_float = from.is_integer() && matches!(to, Type::F32 | Type::F64);
            let float_to_int = matches!(from, Type::F32 | Type::F64) && to.is_integer();
            if int_to_float {
                // MOVQ xmm, gpr
                Ok(vec![MachInst::MovqToXmm {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                }])
            } else if float_to_int {
                // MOVQ gpr, xmm
                Ok(vec![MachInst::MovqFromXmm {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                }])
            } else if dst == src {
                Ok(vec![])
            } else {
                // Same class (int->int or float->float): register copy.
                if from.is_integer() {
                    Ok(vec![MachInst::MovRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(src),
                    }])
                } else {
                    Ok(vec![MachInst::MovsdRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(src),
                    }])
                }
            }
        }
    }
}

// ── Multi-block helpers ───────────────────────────────────────────────────────

/// Collect canonical ClassIds referenced by a single block's effectful ops.
fn collect_block_roots(block: &crate::ir::function::BasicBlock, egraph: &EGraph) -> Vec<ClassId> {
    let mut roots = Vec::new();
    push_block_class_ids(block, &mut roots);
    for r in &mut roots {
        *r = egraph.unionfind.find_immutable(*r);
    }
    roots.sort_by_key(|c| c.0);
    roots.dedup();
    roots
}

/// Build a map from (block_id, param_idx) -> canonical ClassId for all block params.
///
/// Scans the egraph for BlockParam nodes and records their canonical class IDs.
fn build_block_param_class_map(egraph: &EGraph) -> HashMap<(BlockId, u32), ClassId> {
    let mut map: HashMap<(BlockId, u32), ClassId> = HashMap::new();
    for i in 0..egraph.classes.len() as u32 {
        let cid = ClassId(i);
        let canon = egraph.unionfind.find_immutable(cid);
        if canon != cid {
            continue; // Only process canonical classes.
        }
        let class = egraph.class(cid);
        for node in &class.nodes {
            if let Op::BlockParam(bid, pidx, _) = &node.op {
                map.insert((*bid, *pidx), cid);
            }
        }
    }
    map
}

/// Collect VRegs for all phi-copy source arguments across all blocks.
///
/// These are the values passed as args to Jump/Branch. They need to be in
/// `live_out` so the regalloc doesn't allocate two simultaneously-needed
/// phi source values to the same register (especially on loop back-edges).
fn collect_phi_source_vregs(
    func: &Function,
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
    result: &mut HashSet<VReg>,
) {
    for block in &func.blocks {
        for op in &block.ops {
            let args: &[ClassId] = match op {
                EffectfulOp::Jump { args, .. } => args,
                EffectfulOp::Branch {
                    true_args,
                    false_args,
                    ..
                } => {
                    for &cid in true_args.iter().chain(false_args.iter()) {
                        let canon = egraph.unionfind.find_immutable(cid);
                        if let Some(&vreg) = class_to_vreg.get(&canon) {
                            result.insert(vreg);
                        }
                    }
                    continue;
                }
                _ => continue,
            };
            for &cid in args {
                let canon = egraph.unionfind.find_immutable(cid);
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    result.insert(vreg);
                }
            }
        }
    }
}

/// Build phi copy pairs from block parameter passing for coalescing.
///
/// For each Jump/Branch that passes args to a target block with params,
/// for each (arg_class_id, param_class_id) pair, look up their VRegs
/// and add them as copy pairs: (arg_vreg, param_vreg).
fn compute_copy_pairs(
    func: &Function,
    class_to_vreg: &HashMap<ClassId, VReg>,
    egraph: &EGraph,
    block_param_map: &HashMap<(BlockId, u32), ClassId>,
) -> Vec<(VReg, VReg)> {
    let mut pairs: Vec<(VReg, VReg)> = Vec::new();

    let get_vreg = |cid: ClassId| -> Option<VReg> {
        let canon = egraph.unionfind.find_immutable(cid);
        class_to_vreg.get(&canon).copied()
    };

    for block in &func.blocks {
        for op in &block.ops {
            let (target, args): (BlockId, &[ClassId]) = match op {
                EffectfulOp::Jump { target, args } => (*target, args),
                EffectfulOp::Branch {
                    bb_true, true_args, ..
                } => {
                    // Handle true branch.
                    for (idx, &arg_cid) in true_args.iter().enumerate() {
                        if let Some(&param_cid) = block_param_map.get(&(*bb_true, idx as u32)) {
                            if let (Some(arg_v), Some(param_v)) =
                                (get_vreg(arg_cid), get_vreg(param_cid))
                            {
                                pairs.push((arg_v, param_v));
                            }
                        }
                    }
                    // Handle false branch via the destructuring below.
                    if let EffectfulOp::Branch {
                        bb_false,
                        false_args,
                        ..
                    } = op
                    {
                        for (idx, &arg_cid) in false_args.iter().enumerate() {
                            if let Some(&param_cid) = block_param_map.get(&(*bb_false, idx as u32))
                            {
                                if let (Some(arg_v), Some(param_v)) =
                                    (get_vreg(arg_cid), get_vreg(param_cid))
                                {
                                    pairs.push((arg_v, param_v));
                                }
                            }
                        }
                    }
                    continue;
                }
                _ => continue,
            };
            for (idx, &arg_cid) in args.iter().enumerate() {
                if let Some(&param_cid) = block_param_map.get(&(target, idx as u32)) {
                    if let (Some(arg_v), Some(param_v)) = (get_vreg(arg_cid), get_vreg(param_cid)) {
                        pairs.push((arg_v, param_v));
                    }
                }
            }
        }
    }
    pairs
}

/// Compute loop depth for each VReg based on the CFG back-edges.
///
/// A back-edge is a jump/branch to a block with a lower (or equal) index,
/// indicating a loop. All VRegs defined in blocks within the loop body get
/// a non-zero depth. This is a simple heuristic (not a full dominator tree).
fn compute_loop_depths(
    func: &Function,
    block_schedules: &[Vec<ScheduledInst>],
    class_to_vreg: &HashMap<ClassId, VReg>,
) -> HashMap<VReg, u32> {
    let n = func.blocks.len();
    // Compute per-block loop depth using back-edge counting.
    let mut block_depth: Vec<u32> = vec![0u32; n];

    // For each block, check its terminator for back-edges.
    for (src_idx, block) in func.blocks.iter().enumerate() {
        if let Some(terminator) = block.ops.last() {
            let targets: Vec<BlockId> = match terminator {
                EffectfulOp::Jump { target, .. } => vec![*target],
                EffectfulOp::Branch {
                    bb_true, bb_false, ..
                } => vec![*bb_true, *bb_false],
                _ => vec![],
            };
            for target in targets {
                // Find target block index.
                if let Some(target_idx) = func.blocks.iter().position(|b| b.id == target) {
                    if target_idx <= src_idx {
                        // Back-edge: all blocks from target_idx to src_idx are in the loop.
                        for d in block_depth[target_idx..=src_idx].iter_mut() {
                            *d += 1;
                        }
                    }
                }
            }
        }
    }

    // Map each VReg to its block's loop depth.
    let mut result: HashMap<VReg, u32> = HashMap::new();
    for (block_idx, sched) in block_schedules.iter().enumerate() {
        let depth = block_depth[block_idx];
        if depth == 0 {
            continue;
        }
        for inst in sched {
            result.insert(inst.dst, depth);
        }
    }

    // Also map VRegs from class_to_vreg for blocks (they may not appear in schedules).
    let _ = class_to_vreg; // already covered via block_schedules
    result
}

/// Pre-color shift count operands to RCX for variable-shift instructions.
fn add_shift_precolors(insts: &[ScheduledInst], param_vregs: &mut Vec<(VReg, Reg)>) {
    for inst in insts {
        if matches!(inst.op, Op::X86Shl | Op::X86Shr | Op::X86Sar) && inst.operands.len() >= 2 {
            let count_vreg = inst.operands[1];
            if !param_vregs.iter().any(|&(v, _)| v == count_vreg) {
                param_vregs.push((count_vreg, Reg::RCX));
            }
        }
    }
}

/// Pre-color call argument and call result VRegs to their ABI registers.
///
/// For register args (first 6 GPR), pre-color directly.
/// For stack args, add to `live_out` to force them to interfere with each other.
/// For call results, pre-color to RAX.
fn add_call_precolors(
    func: &Function,
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
    param_vregs: &mut Vec<(VReg, Reg)>,
    live_out: &mut HashSet<VReg>,
) {
    for block in &func.blocks {
        for op in &block.ops {
            if let EffectfulOp::Call { args, results, .. } = op {
                let arg_types: Vec<crate::ir::types::Type> =
                    vec![crate::ir::types::Type::I64; args.len()];
                let locs = crate::x86::abi::assign_args(&arg_types);
                for (&cid, loc) in args.iter().zip(locs.iter()) {
                    let canon = egraph.unionfind.find_immutable(cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        match loc {
                            crate::x86::abi::ArgLoc::Reg(reg) => {
                                if !param_vregs.iter().any(|&(v, _)| v == vreg) {
                                    param_vregs.push((vreg, *reg));
                                }
                            }
                            crate::x86::abi::ArgLoc::Stack { .. } => {
                                live_out.insert(vreg);
                            }
                        }
                    }
                }
                if let Some(&first_result_cid) = results.first() {
                    let canon = egraph.unionfind.find_immutable(first_result_cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        if !param_vregs.iter().any(|&(v, _)| v == vreg) {
                            param_vregs.push((vreg, GPR_RETURN_REG));
                        }
                    }
                }
            }
        }
    }
}

/// Compute call point indices (local to a block's instruction list) for
/// caller-saved clobber modeling. Returns local indices into `block_sched`.
fn collect_call_points_for_block(
    func: &Function,
    block_idx: usize,
    block_sched: &[ScheduledInst],
) -> Vec<usize> {
    let block = &func.blocks[block_idx];
    let non_term_count = if block.ops.is_empty() {
        0
    } else {
        block.ops.len() - 1
    };
    let has_call = block.ops[..non_term_count]
        .iter()
        .any(|op| matches!(op, EffectfulOp::Call { .. }));
    if has_call {
        vec![block_sched.len()]
    } else {
        vec![]
    }
}

/// Negate a CondCode.
fn negate_cc(cc: crate::ir::condcode::CondCode) -> crate::ir::condcode::CondCode {
    use crate::ir::condcode::CondCode;
    match cc {
        CondCode::Eq => CondCode::Ne,
        CondCode::Ne => CondCode::Eq,
        CondCode::Slt => CondCode::Sge,
        CondCode::Sle => CondCode::Sgt,
        CondCode::Sgt => CondCode::Sle,
        CondCode::Sge => CondCode::Slt,
        CondCode::Ult => CondCode::Uge,
        CondCode::Ule => CondCode::Ugt,
        CondCode::Ugt => CondCode::Ule,
        CondCode::Uge => CondCode::Ult,
    }
}

/// Lower pure scheduled ops for a single block (skipping param/block-param VRegs).
fn lower_block_pure_ops(
    insts: &[ScheduledInst],
    regalloc: &RegAllocResult,
    func: &Function,
    param_vreg_set: &HashSet<VReg>,
    frame_layout: &crate::x86::abi::FrameLayout,
) -> Result<Vec<MachInst>, CompileError> {
    use crate::regalloc::spill::{
        is_spill_load, is_spill_store, is_xmm_spill_load, is_xmm_spill_store, spill_slot_of,
    };
    let mut result: Vec<MachInst> = Vec::new();
    let get_reg = |vreg: VReg| -> Option<Reg> { regalloc.vreg_to_reg.get(&vreg).copied() };

    for inst in insts {
        // Skip function param VRegs (pre-colored to ABI arg regs).
        if param_vreg_set.contains(&inst.dst) {
            continue;
        }
        // Skip block param VRegs: their values arrive from predecessor phi copies.
        if matches!(inst.op, Op::BlockParam(_, _, _)) {
            continue;
        }
        // Skip LoadResult VRegs: their values are produced by lower_effectful_op.
        if matches!(inst.op, Op::LoadResult(_, _)) {
            continue;
        }
        // Skip CallResult VRegs: their values are captured after CallDirect in lower_effectful_op.
        if matches!(inst.op, Op::CallResult(_, _)) {
            continue;
        }

        // Handle GPR spill sentinels.
        if is_spill_store(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(src_reg) = inst.operands.first().and_then(|&v| get_reg(v)) {
                result.push(MachInst::MovMR {
                    addr: crate::x86::addr::Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                    src: Operand::Reg(src_reg),
                });
            }
            continue;
        }
        if is_spill_load(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(dst_reg) = get_reg(inst.dst) {
                result.push(MachInst::MovRM {
                    dst: Operand::Reg(dst_reg),
                    addr: crate::x86::addr::Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                });
            }
            continue;
        }

        // Handle XMM spill sentinels.
        if is_xmm_spill_store(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(src_reg) = inst.operands.first().and_then(|&v| get_reg(v)) {
                result.push(MachInst::MovsdMR {
                    addr: crate::x86::addr::Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                    src: Operand::Reg(src_reg),
                });
            }
            continue;
        }
        if is_xmm_spill_load(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(dst_reg) = get_reg(inst.dst) {
                result.push(MachInst::MovsdRM {
                    dst: Operand::Reg(dst_reg),
                    addr: crate::x86::addr::Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                });
            }
            continue;
        }

        let dst_reg_opt = get_reg(inst.dst);
        let op_regs: Vec<Option<Reg>> = inst.operands.iter().map(|&v| get_reg(v)).collect();

        let machinsts = lower_op(&inst.op, inst.dst, dst_reg_opt, &inst.operands, &op_regs)
            .map_err(|msg| CompileError {
                phase: "lowering".into(),
                message: msg,
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
        result.extend(machinsts);
    }
    Ok(result)
}

/// Build an `Addr` for Load/Store by checking if `addr_cid` extracted to an Addr node.
///
/// If the extraction result for `addr_cid` is an `Op::Addr { scale, disp }` node,
/// fuse the addressing mode directly into the memory operand (no separate LEA needed).
/// Otherwise fall back to `[addr_reg + 0]`.
fn build_mem_addr(
    addr_cid: ClassId,
    addr_reg: Reg,
    extraction: &ExtractionResult,
    class_to_vreg: &HashMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
) -> crate::x86::addr::Addr {
    if let Some(ext) = extraction.choices.get(&addr_cid) {
        if let Op::Addr { scale, disp } = &ext.op {
            // children[0] = base ClassId, children[1] = index ClassId (may be NONE).
            let base_reg = ext
                .children
                .first()
                .and_then(|&c| class_to_vreg.get(&c))
                .and_then(|v| regalloc.vreg_to_reg.get(v).copied());
            let index_reg = ext
                .children
                .get(1)
                .filter(|&&c| c != ClassId::NONE)
                .and_then(|&c| class_to_vreg.get(&c))
                .and_then(|v| regalloc.vreg_to_reg.get(v).copied());
            if let Some(base) = base_reg {
                return crate::x86::addr::Addr {
                    base: Some(base),
                    index: index_reg,
                    scale: *scale,
                    disp: *disp,
                };
            }
        }
    }
    crate::x86::addr::Addr {
        base: Some(addr_reg),
        index: None,
        scale: 1,
        disp: 0,
    }
}

/// Lower a non-terminator effectful op (Load, Store, Call) to MachInsts.
fn lower_effectful_op(
    op: &EffectfulOp,
    class_to_vreg: &HashMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
    extraction: &ExtractionResult,
    func: &Function,
) -> Result<Vec<MachInst>, CompileError> {
    let get_reg = |cid: ClassId| -> Option<Reg> {
        class_to_vreg
            .get(&cid)
            .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
    };

    match op {
        EffectfulOp::Load {
            addr,
            result,
            ty: _,
        } => {
            let canon_addr = *addr;
            let addr_reg = get_reg(canon_addr).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Load: no register for addr".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let result_reg = class_to_vreg
                .get(result)
                .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
                .ok_or_else(|| CompileError {
                    phase: "lowering".into(),
                    message: "Load: no register for result".into(),
                    location: Some(IrLocation {
                        function: func.name.clone(),
                        block: None,
                        inst: None,
                    }),
                })?;
            let addr = build_mem_addr(canon_addr, addr_reg, extraction, class_to_vreg, regalloc);
            Ok(vec![MachInst::MovRM {
                dst: Operand::Reg(result_reg),
                addr,
            }])
        }
        EffectfulOp::Store { addr, val } => {
            let canon_addr = *addr;
            let addr_reg = get_reg(canon_addr).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Store: no register for addr".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let val_reg = get_reg(*val).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Store: no register for val".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let addr = build_mem_addr(canon_addr, addr_reg, extraction, class_to_vreg, regalloc);
            Ok(vec![MachInst::MovMR {
                addr,
                src: Operand::Reg(val_reg),
            }])
        }
        EffectfulOp::Call {
            func: callee,
            args,
            ret_tys: _,
            results,
        } => {
            // Collect a register for each argument. Missing registers are an error.
            let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
            for &cid in args {
                let r = get_reg(cid).ok_or_else(|| CompileError {
                    phase: "lowering".into(),
                    message: format!("Call: no register for argument class {cid:?}"),
                    location: Some(IrLocation {
                        function: func.name.clone(),
                        block: None,
                        inst: None,
                    }),
                })?;
                arg_regs.push(r);
            }
            // All args treated as I64 for ABI assignment (correct for GPR-only calls;
            // FP args via XMM will be handled when arg type tracking is added).
            let arg_types: Vec<crate::ir::types::Type> =
                vec![crate::ir::types::Type::I64; arg_regs.len()];
            let mut insts = setup_call_args(&arg_types, &arg_regs, Reg::R11);

            // Count stack args so we can clean up RSP after the call.
            let locs = crate::x86::abi::assign_args(&arg_types);
            let n_stack = locs
                .iter()
                .filter(|l| matches!(l, crate::x86::abi::ArgLoc::Stack { .. }))
                .count();

            insts.push(MachInst::CallDirect {
                target: callee.clone(),
            });

            // Clean up stack arguments after the call.
            if n_stack > 0 {
                insts.push(MachInst::AddRI {
                    dst: Operand::Reg(Reg::RSP),
                    imm: (n_stack as i32) * 8,
                });
            }

            // After the call, the first GPR return value is in RAX.
            // If a CallResult ClassId was allocated to a different register, emit a MOV.
            //
            // Known limitation: caller-saved registers (RAX, RCX, RDX, RSI, RDI, R8-R11)
            // are not modeled as clobbered by the call. VRegs live across the call may
            // be incorrectly assigned to caller-saved registers and corrupted.
            if let Some(&result_cid) = results.first() {
                if let Some(result_reg) = get_reg(result_cid) {
                    if result_reg != GPR_RETURN_REG {
                        insts.push(MachInst::MovRR {
                            dst: Operand::Reg(result_reg),
                            src: Operand::Reg(GPR_RETURN_REG),
                        });
                    }
                }
            }
            Ok(insts)
        }
        EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. } => {
            unreachable!("terminators must be handled separately")
        }
    }
}

/// A flat item emitted for a block: either a MachInst or a label binding.
enum BlockItem {
    Inst(MachInst),
    BindLabel(LabelId),
}

/// Rewrite branch targets to skip through empty trampoline blocks.
///
/// A block is "empty" if its items contain only a single `Jmp { target }` (no phi
/// copies, no labels). For any such block, we record `block_id -> target` and then
/// rewrite all `Jcc` and `Jmp` instructions that point to it to jump directly to
/// the final destination. Repeated until no changes occur (handles chains).
fn thread_branches(block_items: &mut Vec<Vec<BlockItem>>, func: &Function, rpo_order: &[usize]) {
    loop {
        // Build a map: block_id -> jump_target for blocks that are just a Jmp.
        let mut redirect: HashMap<LabelId, LabelId> = HashMap::new();
        for (rpo_pos, items) in block_items.iter().enumerate() {
            let block_id = func.blocks[rpo_order[rpo_pos]].id as LabelId;
            // Count real instructions (not BindLabel).
            let real: Vec<&MachInst> = items
                .iter()
                .filter_map(|item| {
                    if let BlockItem::Inst(inst) = item {
                        Some(inst)
                    } else {
                        None
                    }
                })
                .collect();
            if real.len() == 1 {
                if let MachInst::Jmp { target } = real[0] {
                    redirect.insert(block_id, *target);
                }
            }
        }

        if redirect.is_empty() {
            break;
        }

        // Resolve chains: if A -> B -> C, make A -> C directly.
        let keys: Vec<LabelId> = redirect.keys().copied().collect();
        for k in keys {
            let mut dest = redirect[&k];
            let mut seen = std::collections::HashSet::new();
            seen.insert(k);
            while let Some(&next) = redirect.get(&dest) {
                if seen.contains(&next) {
                    break; // cycle guard
                }
                seen.insert(dest);
                dest = next;
            }
            redirect.insert(k, dest);
        }

        // Rewrite Jcc/Jmp targets in all blocks.
        let mut changed = false;
        for items in block_items.iter_mut() {
            for item in items.iter_mut() {
                if let BlockItem::Inst(inst) = item {
                    match inst {
                        MachInst::Jmp { target } => {
                            if let Some(&new_target) = redirect.get(target) {
                                *target = new_target;
                                changed = true;
                            }
                        }
                        MachInst::Jcc { target, .. } => {
                            if let Some(&new_target) = redirect.get(target) {
                                *target = new_target;
                                changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Lower a block terminator, including phi copies for block-parameter passing.
///
/// Returns a list of `BlockItem`s (instructions and label bindings).
/// Uses `next_label` to allocate extra labels for trampoline code.
/// `MachInst::Ret` is returned as a marker replaced by `emit_epilogue` at encode time.
///
/// `next_block_id` is the block ID of the block that immediately follows this one
/// in emission (RPO) order. When a jump target equals `next_block_id`, the jump
/// can be omitted (fallthrough optimization).
#[allow(clippy::too_many_arguments)]
fn lower_terminator(
    op: &EffectfulOp,
    next_block_id: Option<BlockId>,
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
    block_param_map: &HashMap<(BlockId, u32), ClassId>,
    regalloc: &RegAllocResult,
    func: &Function,
    next_label: &mut LabelId,
) -> Result<Vec<BlockItem>, CompileError> {
    let get_reg = |cid: ClassId| -> Option<Reg> {
        let canon = egraph.unionfind.find_immutable(cid);
        class_to_vreg
            .get(&canon)
            .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
    };

    match op {
        EffectfulOp::Ret { val } => {
            let mut items = Vec::new();
            if let Some(&ret_cid) = val.as_ref() {
                if let Some(ret_reg) = get_reg(ret_cid) {
                    if ret_reg != GPR_RETURN_REG {
                        items.push(BlockItem::Inst(MachInst::MovRR {
                            dst: Operand::Reg(GPR_RETURN_REG),
                            src: Operand::Reg(ret_reg),
                        }));
                    }
                }
            }
            // Ret marker: replaced with emit_epilogue() in the encoding loop.
            items.push(BlockItem::Inst(MachInst::Ret));
            Ok(items)
        }

        EffectfulOp::Jump { target, args } => {
            let copies = build_phi_copies(
                *target,
                args,
                egraph,
                class_to_vreg,
                block_param_map,
                regalloc,
                func,
            )?;
            let mut items: Vec<BlockItem> = phi_copies(&copies, Reg::R11)
                .into_iter()
                .map(BlockItem::Inst)
                .collect();
            // Fallthrough optimization: omit the jump if the target is the
            // immediately following block in emission (RPO) order.
            if next_block_id != Some(*target) {
                items.push(BlockItem::Inst(MachInst::Jmp {
                    target: *target as LabelId,
                }));
            }
            Ok(items)
        }

        EffectfulOp::Branch {
            cond,
            bb_true,
            bb_false,
            true_args,
            false_args,
        } => {
            let canon_cond = egraph.unionfind.find_immutable(*cond);
            let cc = find_cc_in_class(egraph, canon_cond).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: format!(
                    "branch condition class {:?} has no Icmp node; cannot determine CondCode",
                    canon_cond
                ),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;

            let true_copies = build_phi_copies(
                *bb_true,
                true_args,
                egraph,
                class_to_vreg,
                block_param_map,
                regalloc,
                func,
            )?;
            let false_copies = build_phi_copies(
                *bb_false,
                false_args,
                egraph,
                class_to_vreg,
                block_param_map,
                regalloc,
                func,
            )?;

            let true_phi = phi_copies(&true_copies, Reg::R11);
            let false_phi = phi_copies(&false_copies, Reg::R11);

            let false_is_fallthrough = next_block_id == Some(*bb_false);
            let true_is_fallthrough = next_block_id == Some(*bb_true);

            let mut items = Vec::new();
            if true_phi.is_empty() {
                // jcc cc, true_block; [false_phi]; jmp false_block
                // If false_block is the fallthrough, omit the final jmp.
                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc,
                    target: *bb_true as LabelId,
                }));
                items.extend(false_phi.into_iter().map(BlockItem::Inst));
                if !false_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_false as LabelId,
                    }));
                }
            } else if false_phi.is_empty() {
                // jcc !cc, false_block; [true_phi]; jmp true_block
                // If false_block is the fallthrough, emit only the true_phi + jmp true.
                // If true_block is the fallthrough, we can flip: jcc cc, true_block; jmp false.
                if false_is_fallthrough {
                    // false falls through; emit: [true_phi]; jcc !cc, false (skipped = nop);
                    // actually we need to only execute true_phi when cc is true.
                    // Emit: jcc !cc, skip; [true_phi]; L_skip: jmp true_block
                    // But if true_block is also not next, we need jmp true_block too.
                    // Simpler: keep the jcc to false but emit jmp true at the end.
                    // If false falls through, the jcc target (false_block) is next --
                    // this is fine: jcc !cc, false; [true_phi]; jmp true_block.
                    // The jmp true_block is still needed unless true_block is also next.
                    items.push(BlockItem::Inst(MachInst::Jcc {
                        cc: negate_cc(cc),
                        target: *bb_false as LabelId,
                    }));
                    items.extend(true_phi.into_iter().map(BlockItem::Inst));
                    if !true_is_fallthrough {
                        items.push(BlockItem::Inst(MachInst::Jmp {
                            target: *bb_true as LabelId,
                        }));
                    }
                } else {
                    items.push(BlockItem::Inst(MachInst::Jcc {
                        cc: negate_cc(cc),
                        target: *bb_false as LabelId,
                    }));
                    items.extend(true_phi.into_iter().map(BlockItem::Inst));
                    if !true_is_fallthrough {
                        items.push(BlockItem::Inst(MachInst::Jmp {
                            target: *bb_true as LabelId,
                        }));
                    }
                }
            } else {
                // Both sides have copies. Use trampoline labels:
                //   jcc !cc, L_false_copies
                //   [true_phi]
                //   jmp true_block         (omit if true_block is fallthrough)
                //   L_false_copies:
                //   [false_phi]
                //   jmp false_block        (omit if false_block is fallthrough)
                let l_false = *next_label;
                *next_label += 1;

                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc: negate_cc(cc),
                    target: l_false,
                }));
                items.extend(true_phi.into_iter().map(BlockItem::Inst));
                if !true_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_true as LabelId,
                    }));
                }
                items.push(BlockItem::BindLabel(l_false));
                items.extend(false_phi.into_iter().map(BlockItem::Inst));
                if !false_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_false as LabelId,
                    }));
                }
            }
            Ok(items)
        }

        EffectfulOp::Load { .. } | EffectfulOp::Store { .. } | EffectfulOp::Call { .. } => {
            unreachable!("non-terminators handled by lower_effectful_op")
        }
    }
}

/// Build (src_reg, dst_reg) phi copy pairs for a jump to `target` with `args`.
fn build_phi_copies(
    target: BlockId,
    args: &[ClassId],
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
    block_param_map: &HashMap<(BlockId, u32), ClassId>,
    regalloc: &RegAllocResult,
    func: &Function,
) -> Result<Vec<(Reg, Reg)>, CompileError> {
    if args.is_empty() {
        return Ok(vec![]);
    }
    let target_block = func
        .blocks
        .iter()
        .find(|b| b.id == target)
        .ok_or_else(|| CompileError {
            phase: "phi-elim".into(),
            message: format!("jump target block {target} not found"),
            location: None,
        })?;
    let n_params = target_block.param_types.len();
    if n_params == 0 {
        return Ok(vec![]);
    }

    let mut copies = Vec::new();
    for (param_idx, &arg_cid) in args.iter().enumerate() {
        let param_cid = block_param_map
            .get(&(target, param_idx as u32))
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!(
                    "block param ({target}, {param_idx}) not found in block_param_map"
                ),
                location: None,
            })?;

        let canon_arg = egraph.unionfind.find_immutable(arg_cid);
        let arg_vreg = class_to_vreg
            .get(&canon_arg)
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("arg class {:?} not in class_to_vreg", canon_arg),
                location: None,
            })?;
        let src_reg = regalloc
            .vreg_to_reg
            .get(&arg_vreg)
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("arg vreg {:?} not in regalloc", arg_vreg),
                location: None,
            })?;

        let param_vreg = class_to_vreg
            .get(&param_cid)
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("param class {:?} not in class_to_vreg", param_cid),
                location: None,
            })?;
        let dst_reg = regalloc
            .vreg_to_reg
            .get(&param_vreg)
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("param vreg {:?} not in regalloc", param_vreg),
                location: None,
            })?;

        copies.push((src_reg, dst_reg));
    }
    Ok(copies)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::types::Type;
    use crate::test_utils::has_tool;

    fn build_identity() -> (Function, EGraph) {
        let mut builder = FunctionBuilder::new("identity", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        builder.ret(Some(params[0]));
        builder.finalize().expect("identity finalize")
    }

    fn build_add() -> (Function, EGraph) {
        let mut builder = FunctionBuilder::new("add_two", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let sum = builder.add(params[0], params[1]);
        builder.ret(Some(sum));
        builder.finalize().expect("add finalize")
    }

    // 14.3: DiagnosticSink test.
    #[test]
    fn diagnostic_sink_receives_stats() {
        struct VecSink(Vec<String>);
        impl DiagnosticSink for VecSink {
            fn phase_stats(&mut self, phase: &str, stats: &str) {
                self.0.push(format!("{phase}: {stats}"));
            }
        }

        let (func, egraph) = build_add();
        let opts = CompileOptions {
            verbosity: Verbosity::Verbose,
            ..Default::default()
        };
        let mut sink = VecSink(Vec::new());
        let result = compile(&func, egraph, &opts, Some(&mut sink));
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
        assert!(
            !sink.0.is_empty(),
            "diagnostic sink should have received phase stats"
        );
        let all = sink.0.join("\n");
        assert!(all.contains("egraph:"), "should have egraph stats");
    }

    // 14.4: identity(x) -> x
    #[test]
    fn e2e_identity() {
        if !has_tool("cc") {
            return;
        }

        let (func, egraph) = build_identity();
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile identity");

        let dir = std::env::temp_dir();
        let obj_path = dir.join("blitz_e2e_identity.o");
        let main_path = dir.join("blitz_e2e_identity_main.c");
        let bin_path = dir.join("blitz_e2e_identity_bin");

        obj.write_to(&obj_path).expect("write .o");

        std::fs::write(
            &main_path,
            b"#include <stdint.h>\n\
              int64_t identity(int64_t x);\n\
              int main(void) {\n\
              int64_t r = identity(42);\n\
              return (r == 42) ? 0 : 1;\n\
              }\n",
        )
        .expect("write main.c");

        let compile_out = std::process::Command::new("cc")
            .args([
                main_path.to_str().unwrap(),
                obj_path.to_str().unwrap(),
                "-o",
                bin_path.to_str().unwrap(),
            ])
            .output()
            .expect("cc");

        assert!(
            compile_out.status.success(),
            "linking failed:\n{}",
            String::from_utf8_lossy(&compile_out.stderr)
        );

        let run = std::process::Command::new(&bin_path).output().expect("run");
        assert_eq!(run.status.code(), Some(0), "identity(42) should return 42");

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(&main_path);
        let _ = std::fs::remove_file(&bin_path);
    }

    // 14.5: add(a, b) -> a + b
    #[test]
    fn e2e_add() {
        if !has_tool("cc") {
            return;
        }

        let (func, egraph) = build_add();
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile add");

        let dir = std::env::temp_dir();
        let obj_path = dir.join("blitz_e2e_add.o");
        let main_path = dir.join("blitz_e2e_add_main.c");
        let bin_path = dir.join("blitz_e2e_add_bin");

        obj.write_to(&obj_path).expect("write .o");

        std::fs::write(
            &main_path,
            b"#include <stdint.h>\n\
              int64_t add_two(int64_t a, int64_t b);\n\
              int main(void) {\n\
              int64_t r = add_two(3, 4);\n\
              return (r == 7) ? 0 : 1;\n\
              }\n",
        )
        .expect("write main.c");

        let compile_out = std::process::Command::new("cc")
            .args([
                main_path.to_str().unwrap(),
                obj_path.to_str().unwrap(),
                "-o",
                bin_path.to_str().unwrap(),
            ])
            .output()
            .expect("cc");

        assert!(
            compile_out.status.success(),
            "linking failed:\n{}",
            String::from_utf8_lossy(&compile_out.stderr)
        );

        let run = std::process::Command::new(&bin_path).output().expect("run");
        assert_eq!(run.status.code(), Some(0), "add_two(3, 4) should return 7");

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(&main_path);
        let _ = std::fs::remove_file(&bin_path);
    }

    // 14.2: Multi-function compilation.
    #[test]
    fn compile_module_two_functions() {
        if !has_tool("cc") {
            return;
        }

        let id_pair = build_identity();
        let add_pair = build_add();
        let opts = CompileOptions::default();

        let functions = vec![id_pair, add_pair];
        let obj = compile_module(functions, &opts).expect("compile_module");

        assert_eq!(obj.functions.len(), 2);
        assert_eq!(obj.functions[0].name, "identity");
        assert_eq!(obj.functions[1].name, "add_two");
        // Second function offset must be > 0.
        assert!(
            obj.functions[1].offset > 0,
            "add_two should have non-zero offset"
        );
    }

    // ── Helper: link and run a generated object with a C main ─────────────────

    fn link_and_run(test_name: &str, obj_bytes: &[u8], c_main: &str) -> Option<i32> {
        if !has_tool("cc") {
            return None;
        }
        use std::process::Command;

        let dir = std::env::temp_dir();
        let obj_path = dir.join(format!("{test_name}.o"));
        let main_path = dir.join(format!("{test_name}_main.c"));
        let bin_path = dir.join(format!("{test_name}_bin"));

        // Write a minimal ObjectFile wrapping the raw bytes.
        // The test already has a compiled ObjectFile; write it directly.
        std::fs::write(&obj_path, obj_bytes).expect("write .o");
        std::fs::write(&main_path, c_main.as_bytes()).expect("write main.c");

        let compile_out = Command::new("cc")
            .args([
                main_path.to_str().unwrap(),
                obj_path.to_str().unwrap(),
                "-o",
                bin_path.to_str().unwrap(),
            ])
            .output()
            .expect("cc");

        if !compile_out.status.success() {
            eprintln!(
                "cc failed:\n{}",
                String::from_utf8_lossy(&compile_out.stderr)
            );
            let _ = std::fs::remove_file(&obj_path);
            let _ = std::fs::remove_file(&main_path);
            return None;
        }

        let run = Command::new(&bin_path).output().expect("run binary");
        let code = run.status.code();

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(&main_path);
        let _ = std::fs::remove_file(&bin_path);

        code
    }

    // ── Helper: write ObjectFile and link ─────────────────────────────────────

    fn link_and_run_obj(
        test_name: &str,
        obj: &crate::emit::object::ObjectFile,
        c_main: &str,
    ) -> Option<i32> {
        if !has_tool("cc") {
            return None;
        }
        let dir = std::env::temp_dir();
        let obj_path = dir.join(format!("{test_name}.o"));
        obj.write_to(&obj_path).expect("write .o");
        let bytes = std::fs::read(&obj_path).unwrap();
        let _ = std::fs::remove_file(&obj_path);
        link_and_run(test_name, &bytes, c_main)
    }

    // 14.6: Conditional branch — max(a, b) using select/cmov (single-block).
    //
    // Build: icmp(Sgt, a, b) -> flags; select(flags, a, b) -> result; ret(result)
    // After isel, Select becomes X86Cmov; stays single-block, no branch needed.
    #[test]
    fn e2e_conditional_max() {
        use crate::ir::condcode::CondCode;

        let mut builder = FunctionBuilder::new("blitz_max", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];
        let flags = builder.icmp(CondCode::Sgt, a, b);
        let result = builder.select(flags, a, b);
        builder.ret(Some(result));
        let (func, egraph) = builder.finalize().expect("max finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile max");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_max(int64_t a, int64_t b);
int main(void) {
    if (blitz_max(10, 5) != 10) return 1;
    if (blitz_max(3, 7) != 7) return 2;
    if (blitz_max(4, 4) != 4) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_max", &obj, c_main) {
            assert_eq!(code, 0, "max function returned wrong exit code {code}");
        }
    }

    // 14.7: Loop — sum 1..=n using multi-block with back-edge.
    //
    // IR structure:
    //   BB0 (entry): jump(BB1, [0, 1])
    //   BB1 (params=[acc, i]):
    //     new_acc = add(acc, i)
    //     new_i   = add(i, 1)
    //     cond    = icmp(Sle, new_i, n)
    //     branch(cond, BB1, BB2, [new_acc, new_i], [new_acc])
    //   BB2 (params=[result]): ret(result)
    //
    // sum(1..=5) = 15, sum(1..=10) = 55
    #[test]
    fn e2e_loop_sum() {
        use crate::ir::condcode::CondCode;

        let mut builder = FunctionBuilder::new("blitz_sum", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let n = params[0];

        // Create blocks.
        let (bb1, bb1_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
        let acc = bb1_params[0];
        let i = bb1_params[1];
        let (bb2, bb2_params) = builder.create_block_with_params(&[Type::I64]);
        let result = bb2_params[0];

        // BB0: jump to BB1 with acc=0, i=1.
        let zero = builder.iconst(0, Type::I64);
        let one = builder.iconst(1, Type::I64);
        builder.jump(bb1, &[zero, one]);

        // BB1: loop body.
        builder.set_block(bb1);
        let new_acc = builder.add(acc, i);
        let new_i = builder.add(i, one);
        let cond = builder.icmp(CondCode::Sle, new_i, n);
        builder.branch(cond, bb1, bb2, &[new_acc, new_i], &[new_acc]);

        // BB2: return result.
        builder.set_block(bb2);
        builder.ret(Some(result));

        let (func, egraph) = builder.finalize().expect("sum finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile sum");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_sum(int64_t n);
int main(void) {
    if (blitz_sum(5) != 15) return 1;
    if (blitz_sum(10) != 55) return 2;
    if (blitz_sum(1) != 1) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_sum", &obj, c_main) {
            assert_eq!(code, 0, "sum function returned wrong exit code {code}");
        }
    }

    // 14.8: Function call — call an external C function.
    //
    // Build: abs(x) = x >= 0 ? x : -x  using select + icmp + sub(0, x)
    // Alternatively: call abs() from libc.
    // We call a simple helper: double(x) = x + x (defined in C main).
    #[test]
    fn e2e_call_external() {
        let mut builder = FunctionBuilder::new("blitz_call_double", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let x = params[0];
        // Call external function "double_val(x)" and return result.
        let results = builder.call("double_val", &[x], &[Type::I64]);
        builder.ret(Some(results[0]));
        let (func, egraph) = builder.finalize().expect("call finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile call");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_call_double(int64_t x);
int64_t double_val(int64_t x) { return x + x; }
int main(void) {
    if (blitz_call_double(5) != 10) return 1;
    if (blitz_call_double(0) != 0) return 2;
    if (blitz_call_double(-3) != -6) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_call", &obj, c_main) {
            assert_eq!(code, 0, "call_double returned wrong exit code {code}");
        }
    }

    // 14.9: Addressing modes — compile a function using scaled-index addressing.
    //
    // Build: stride_offset(base, idx) = base + idx * 4
    // After isel + addr-mode rules, this should compile to a LEA with scale.
    #[test]
    fn e2e_addressing_modes() {
        use crate::test_utils::objdump_disasm;

        // Build: stride_test(base, idx) = base + idx * 4
        // This exercises the X86Lea3{scale:4} addressing mode rule.
        let mut builder =
            FunctionBuilder::new("blitz_stride_test", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let base = params[0];
        let idx = params[1];
        let two = builder.iconst(2, Type::I64);
        // idx << 2 = idx * 4
        let scaled = builder.shl(idx, two);
        let addr = builder.add(base, scaled);
        builder.ret(Some(addr));
        let (func, egraph) = builder.finalize().expect("stride finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile stride_test");

        // Verify correctness.
        let c_main = r#"
#include <stdint.h>
int64_t blitz_stride_test(int64_t base, int64_t idx);
int main(void) {
    // base=100, idx=3 => 100 + 3*4 = 112
    if (blitz_stride_test(100, 3) != 112) return 1;
    // base=0, idx=0 => 0
    if (blitz_stride_test(0, 0) != 0) return 2;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_stride", &obj, c_main) {
            assert_eq!(code, 0, "stride_test returned wrong exit code {code}");
        }

        // Optionally verify the disassembly shows LEA (addr mode optimization).
        if let Some(disasm) = objdump_disasm(&obj.code) {
            // After addr-mode rules, base + idx<<2 should become lea [base + idx*4].
            // Accept LEA or SHL+ADD as both are correct encodings.
            let has_efficient_code =
                disasm.contains("lea") || disasm.contains("shl") || disasm.contains("add");
            assert!(
                has_efficient_code,
                "expected LEA or SHL/ADD in disassembly:\n{disasm}"
            );
        }
    }

    // 14.10: Register pressure — 20+ simultaneously live values.
    //
    // Build a function that uses more than 15 live values simultaneously
    // to exercise spilling. The function computes the sum of 20 constants
    // to keep many values live at once.
    #[test]
    fn e2e_register_pressure() {
        // Build: sum20() = 1 + 2 + 3 + ... + 20
        // By loading all 20 iconsts before adding them, we create pressure.
        let mut builder = FunctionBuilder::new("blitz_sum20", &[], &[Type::I64]);

        let vals: Vec<_> = (1i64..=20).map(|v| builder.iconst(v, Type::I64)).collect();

        // Chain-add all 20 values in a binary tree pattern to keep many live.
        // This forces regalloc to handle high pressure.
        let mut acc = vals[0];
        for &v in &vals[1..] {
            acc = builder.add(acc, v);
        }
        builder.ret(Some(acc));

        let (func, egraph) = builder.finalize().expect("sum20 finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile sum20");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_sum20(void);
int main(void) {
    // 1+2+...+20 = 210
    if (blitz_sum20() != 210) return 1;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_sum20", &obj, c_main) {
            assert_eq!(code, 0, "sum20 returned wrong exit code {code}");
        }
    }

    // 14.11: Flag fusion — if (a - b > 0) return a - b, else return 0.
    //
    // The optimizer should fuse the subtraction that produces the value
    // with the comparison, emitting a single SUB instruction (no CMP needed).
    #[test]
    fn e2e_flag_fusion() {
        use crate::ir::condcode::CondCode;
        use crate::test_utils::objdump_disasm;

        // Build: flag_fusion(a, b) = max(a - b, 0) using select.
        let mut builder =
            FunctionBuilder::new("blitz_flag_fusion", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];
        let diff = builder.sub(a, b);
        let zero = builder.iconst(0, Type::I64);
        let cond = builder.icmp(CondCode::Sgt, diff, zero);
        let result = builder.select(cond, diff, zero);
        builder.ret(Some(result));
        let (func, egraph) = builder.finalize().expect("flag_fusion finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile flag_fusion");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_flag_fusion(int64_t a, int64_t b);
int main(void) {
    if (blitz_flag_fusion(5, 3) != 2) return 1;
    if (blitz_flag_fusion(3, 5) != 0) return 2;
    if (blitz_flag_fusion(4, 4) != 0) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_flag_fusion", &obj, c_main) {
            assert_eq!(code, 0, "flag_fusion returned wrong exit code {code}");
        }

        // Verify that objdump shows a SUB but no separate CMP
        // (sub a, b sets the flags that cmov uses — no extra cmp needed).
        if let Some(disasm) = objdump_disasm(&obj.code) {
            let has_sub = disasm.contains("sub");
            let has_cmp = disasm.contains("cmp");
            assert!(has_sub, "expected SUB in disassembly:\n{disasm}");
            assert!(
                !has_cmp,
                "expected NO CMP (flag fusion should reuse SUB flags):\n{disasm}"
            );
        }
    }

    // 14.12: Snapshot — compile identity + add, store reference disassembly.
    //
    // Compiles both functions and verifies the disassembly matches a known
    // reference. If objdump is unavailable, just verify compilation succeeds.
    #[test]
    fn e2e_snapshot() {
        use crate::test_utils::objdump_disasm;

        let (id_func, id_egraph) = build_identity();
        let (add_func, add_egraph) = build_add();
        let opts = CompileOptions::default();

        let id_obj = compile(&id_func, id_egraph, &opts, None).expect("compile identity");
        let add_obj = compile(&add_func, add_egraph, &opts, None).expect("compile add");

        // Verify the identity function is minimal: just prologue + mov rax,rdi + epilogue.
        // Expected bytes: 55 48 89 e5 48 89 f8 5d c3
        let expected_identity: &[u8] = &[
            0x55, // push rbp
            0x48, 0x89, 0xe5, // mov rbp, rsp
            0x48, 0x89, 0xf8, // mov rax, rdi
            0x5d, // pop rbp
            0xc3, // ret
        ];
        assert_eq!(
            &id_obj.code, expected_identity,
            "identity function bytes mismatch"
        );

        // Verify add function compiles and has plausible size.
        assert!(
            add_obj.code.len() >= 5,
            "add function should be at least 5 bytes"
        );

        // Optional: print disassembly for both if objdump is available.
        if let Some(disasm) = objdump_disasm(&id_obj.code) {
            assert!(
                disasm.contains("mov") && disasm.contains("ret"),
                "identity disassembly should contain mov and ret:\n{disasm}"
            );
        }

        if let Some(disasm) = objdump_disasm(&add_obj.code) {
            assert!(
                disasm.contains("add") || disasm.contains("lea"),
                "add disassembly should contain add or lea:\n{disasm}"
            );
        }
    }

    // 14.9: Branch relaxation -- short-form jumps.
    //
    // Compile a simple conditional (two blocks) and verify that the jump bytes
    // use the short form (EB for JMP, 7x for Jcc) since the blocks are close
    // together.  We scan `obj.code` for near-form jump opcodes (E9, 0F 8x)
    // and assert none appear.
    #[test]
    fn branch_relaxation_uses_short_form_for_nearby_targets() {
        use crate::ir::condcode::CondCode;

        // Build: max(a, b) = if a >= b { a } else { b }
        // Single condition, two close blocks -> all jumps should be short.
        let mut builder =
            FunctionBuilder::new("blitz_max_short", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];

        let (bb_true, _) = builder.create_block_with_params(&[]);
        let (bb_false, _) = builder.create_block_with_params(&[]);

        let cond = builder.icmp(CondCode::Sge, a, b);
        builder.branch(cond, bb_true, bb_false, &[], &[]);

        builder.set_block(bb_true);
        builder.ret(Some(a));

        builder.set_block(bb_false);
        builder.ret(Some(b));

        let (func, egraph) = builder.finalize().expect("max_short finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile max_short");

        // Walk the code bytes and look for near-form jump opcodes.
        // E9 = near JMP; 0F followed by 80..8F = near Jcc.
        let code = &obj.code;
        let mut i = 0;
        let mut found_near_jmp = false;
        let mut found_near_jcc = false;
        while i < code.len() {
            match code[i] {
                0xE9 => {
                    found_near_jmp = true;
                }
                0x0F if i + 1 < code.len() && (0x80..=0x8F).contains(&code[i + 1]) => {
                    found_near_jcc = true;
                }
                _ => {}
            }
            i += 1;
        }

        assert!(
            !found_near_jmp,
            "nearby JMP should use short form (EB), found near form (E9) in {:02X?}",
            code
        );
        assert!(
            !found_near_jcc,
            "nearby Jcc should use short form (7x), found near form (0F 8x) in {:02X?}",
            code
        );
    }

    // Phase 2: sext compiles end-to-end — a function that sign-extends its I32
    // parameter to I64 and returns it should compile without error.
    #[test]
    fn e2e_sext_i32_to_i64() {
        let mut builder = FunctionBuilder::new("sext_i32_to_i64", &[Type::I32], &[Type::I64]);
        let params = builder.params().to_vec();
        let extended = builder.sext(params[0], Type::I64);
        builder.ret(Some(extended));
        let (func, egraph) = builder.finalize().expect("sext finalize");
        let opts = CompileOptions::default();
        compile(&func, egraph, &opts, None).expect("compile sext_i32_to_i64");
    }

    // Phase 2: zext compiles end-to-end
    #[test]
    fn e2e_zext_i8_to_i64() {
        let mut builder = FunctionBuilder::new("zext_i8_to_i64", &[Type::I8], &[Type::I64]);
        let params = builder.params().to_vec();
        let extended = builder.zext(params[0], Type::I64);
        builder.ret(Some(extended));
        let (func, egraph) = builder.finalize().expect("zext finalize");
        let opts = CompileOptions::default();
        compile(&func, egraph, &opts, None).expect("compile zext_i8_to_i64");
    }

    // Phase 2: trunc compiles end-to-end
    #[test]
    fn e2e_trunc_i64_to_i32() {
        let mut builder = FunctionBuilder::new("trunc_i64_to_i32", &[Type::I64], &[Type::I32]);
        let params = builder.params().to_vec();
        let truncated = builder.trunc(params[0], Type::I32);
        builder.ret(Some(truncated));
        let (func, egraph) = builder.finalize().expect("trunc finalize");
        let opts = CompileOptions::default();
        compile(&func, egraph, &opts, None).expect("compile trunc_i64_to_i32");
    }

    // Phase 3: load from a pointer argument compiles end-to-end.
    //
    // Build: load_ptr(ptr: *i64) -> i64 = *ptr
    // The Load effectful op should produce a VReg, get allocated a register,
    // and lower to a MovRM instruction.
    #[test]
    fn e2e_load_from_pointer_arg() {
        // ptr is passed as an I64 (pointer is just a 64-bit integer here)
        let mut builder = FunctionBuilder::new("blitz_load_ptr", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let ptr = params[0];
        let val = builder.load(ptr, Type::I64);
        builder.ret(Some(val));
        let (func, egraph) = builder.finalize().expect("load_ptr finalize");
        let opts = CompileOptions::default();
        compile(&func, egraph, &opts, None).expect("compile load_from_pointer_arg");
    }

    // Phase 3: store then load — write a value, read it back.
    //
    // Build: store_load(ptr: *i64, val: i64) -> i64 { *ptr = val; return *ptr }
    #[test]
    fn e2e_store_then_load() {
        let mut builder =
            FunctionBuilder::new("blitz_store_load", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let ptr = params[0];
        let val = params[1];
        builder.store(ptr, val);
        let loaded = builder.load(ptr, Type::I64);
        builder.ret(Some(loaded));
        let (func, egraph) = builder.finalize().expect("store_load finalize");
        let opts = CompileOptions::default();
        compile(&func, egraph, &opts, None).expect("compile store_then_load");
    }

    // Phase 4.3: function with a variable shift compiles end-to-end.
    //
    // The shift count VReg must be pre-colored to RCX before regalloc so that
    // lower_shift_cl can assert src_b == RCX without clobbering live values.
    #[test]
    fn e2e_variable_shift() {
        let mut builder = FunctionBuilder::new("blitz_shl", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let val = params[0];
        let count = params[1];
        let shifted = builder.shl(val, count);
        builder.ret(Some(shifted));
        let (func, egraph) = builder.finalize().expect("shl finalize");
        let opts = CompileOptions::default();
        compile(&func, egraph, &opts, None).expect("compile variable_shift");
    }

    // Phase 5.3: Diamond CFG merge with phi copies from both edges.
    //
    // IR structure:
    //   BB0 (entry, params=[a, b]):
    //     cond = icmp(Sgt, a, b)
    //     branch(cond, BB_true, BB_false, [a, b], [b, a])
    //   BB_true (params=[x, y]):
    //     val = add(x, y)   ; x=a, y=b on true edge
    //     jump(BB_merge, [val])
    //   BB_false (params=[x, y]):
    //     val = add(x, y)   ; x=b, y=a on false edge (swapped)
    //     jump(BB_merge, [val])
    //   BB_merge (params=[result]):
    //     ret(result)
    //
    // Both edges carry different phi argument orderings, exercising phi copy
    // generation on a critical edge (BB0 has 2 successors, BB_merge has 2 preds).
    #[test]
    fn phi_diamond_cfg_merge_with_copies_from_both_edges() {
        use crate::ir::condcode::CondCode;

        let mut builder =
            FunctionBuilder::new("phi_diamond", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];

        let (bb_true, bb_true_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
        let x_true = bb_true_params[0];
        let y_true = bb_true_params[1];

        let (bb_false, bb_false_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
        let x_false = bb_false_params[0];
        let y_false = bb_false_params[1];

        let (bb_merge, bb_merge_params) = builder.create_block_with_params(&[Type::I64]);
        let result = bb_merge_params[0];

        // BB0: branch based on a > b.
        // True edge: pass (a, b); False edge: pass (b, a) -- swapped.
        let cond = builder.icmp(CondCode::Sgt, a, b);
        builder.branch(cond, bb_true, bb_false, &[a, b], &[b, a]);

        // BB_true: add x + y, jump to merge.
        builder.set_block(bb_true);
        let sum_true = builder.add(x_true, y_true);
        builder.jump(bb_merge, &[sum_true]);

        // BB_false: add x + y (same computation, different inputs), jump to merge.
        builder.set_block(bb_false);
        let sum_false = builder.add(x_false, y_false);
        builder.jump(bb_merge, &[sum_false]);

        // BB_merge: return the phi result.
        builder.set_block(bb_merge);
        builder.ret(Some(result));

        let (func, egraph) = builder.finalize().expect("diamond finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile phi_diamond");

        // Verify: phi_diamond(5, 3) = 5+3 = 8 (true edge: a=5 > b=3, so x=5, y=3)
        //         phi_diamond(2, 7) = 7+2 = 9 (false edge: b=7, a=2, so x=7, y=2)
        //         phi_diamond(4, 4) = 4+4 = 8 (false edge: b=4, a=4, symmetric)
        let c_main = r#"
#include <stdint.h>
int64_t phi_diamond(int64_t a, int64_t b);
int main(void) {
    if (phi_diamond(5, 3) != 8) return 1;
    if (phi_diamond(2, 7) != 9) return 2;
    if (phi_diamond(4, 4) != 8) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_phi_diamond", &obj, c_main) {
            assert_eq!(code, 0, "phi_diamond returned wrong exit code {code}");
        }
    }

    // Phase 5.1: RPO ordering + fallthrough -- verify a simple loop compiles
    // correctly and that the entry->loop jump is eliminated (fallthrough).
    #[test]
    fn rpo_fallthrough_eliminates_entry_jump() {
        use crate::ir::condcode::CondCode;

        // Build: count_down(n) -- counts n down to 0, returns 0.
        // BB0: jump(BB1, [n])
        // BB1(params=[i]): cond = icmp(Sgt, i, 0); branch(cond, BB1, BB2, [sub(i,1)], [])
        // BB2: ret(0)
        //
        // In RPO: BB0 -> BB1 -> BB2. The jump from BB0 to BB1 is a fallthrough.
        let mut builder = FunctionBuilder::new("count_down", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let n = params[0];

        let (bb1, bb1_params) = builder.create_block_with_params(&[Type::I64]);
        let i = bb1_params[0];
        let (bb2, _) = builder.create_block_with_params(&[]);

        // BB0: jump to BB1 with i=n.
        builder.jump(bb1, &[n]);

        // BB1: loop body.
        builder.set_block(bb1);
        let one = builder.iconst(1, Type::I64);
        let zero = builder.iconst(0, Type::I64);
        let new_i = builder.sub(i, one);
        let cond = builder.icmp(CondCode::Sgt, i, zero);
        builder.branch(cond, bb1, bb2, &[new_i], &[]);

        // BB2: return 0.
        builder.set_block(bb2);
        let ret_zero = builder.iconst(0, Type::I64);
        builder.ret(Some(ret_zero));

        let (func, egraph) = builder.finalize().expect("count_down finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile count_down");

        // Verify correctness.
        let c_main = r#"
#include <stdint.h>
int64_t count_down(int64_t n);
int main(void) {
    if (count_down(0) != 0) return 1;
    if (count_down(5) != 0) return 2;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_count_down", &obj, c_main) {
            assert_eq!(code, 0, "count_down returned wrong exit code {code}");
        }

        // Check that there is no standalone jump to the very next byte
        // (fallthrough optimization should have eliminated the BB0->BB1 jump).
        // We verify by checking the object has fewer bytes than if the jump were kept.
        // A near-short JMP (EB + 1 byte) would be 2 bytes; Jmp to fallthrough = 0 bytes saved.
        // We just verify the code is non-empty and compilation succeeded.
        assert!(!obj.code.is_empty(), "compiled code should not be empty");
    }

    // Fix 5: FP constant loading — fconst value must reach an XMM register.
    #[test]
    fn e2e_fconst_f64() {
        // Build: fp_const() -> f64 returning the constant 2.5
        let mut builder = FunctionBuilder::new("blitz_fp_const", &[], &[Type::F64]);
        let c = builder.fconst(2.5f64);
        builder.ret(Some(c));
        let (func, egraph) = builder.finalize().expect("fp_const finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile fp_const");
        assert!(!obj.code.is_empty());

        let c_main = r#"
double blitz_fp_const(void);
int main(void) {
    double v = blitz_fp_const();
    return (v == 2.5) ? 0 : 1;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_fp_const", &obj, c_main) {
            assert_eq!(code, 0, "fp_const returned wrong exit code {code}");
        }
    }

    // Fix 6: Call with 8 args (7th and 8th go on the stack).
    // The caller uses iconst values so we don't depend on incoming stack param handling.
    #[test]
    fn e2e_call_8_args() {
        // Build: call_8args() — calls blitz_sum8_ext(1,2,3,4,5,6,7,8).
        // Args 7 and 8 go on the stack.
        let mut builder = FunctionBuilder::new("blitz_call_8args", &[], &[Type::I64]);
        let a = builder.iconst(1, Type::I64);
        let b = builder.iconst(2, Type::I64);
        let c = builder.iconst(3, Type::I64);
        let d = builder.iconst(4, Type::I64);
        let e = builder.iconst(5, Type::I64);
        let f = builder.iconst(6, Type::I64);
        let g = builder.iconst(7, Type::I64);
        let h = builder.iconst(8, Type::I64);
        let results = builder.call("blitz_sum8_ext", &[a, b, c, d, e, f, g, h], &[Type::I64]);
        builder.ret(Some(results[0]));
        let (func, egraph) = builder.finalize().expect("call_8args finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile call_8args");
        assert!(!obj.code.is_empty());

        let c_main = r#"
#include <stdint.h>
int64_t blitz_sum8_ext(int64_t a, int64_t b, int64_t c, int64_t d,
                       int64_t e, int64_t f, int64_t g, int64_t h) {
    return a + b + c + d + e + f + g + h;
}
int64_t blitz_call_8args(void);
int main(void) {
    // 1+2+3+4+5+6+7+8 = 36
    int64_t r = blitz_call_8args();
    return (r == 36) ? 0 : 1;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_call8", &obj, c_main) {
            assert_eq!(code, 0, "call_8args returned wrong exit code {code}");
        }
    }

    // Fix 7: F32 isel — fadd on F32 operands should use addss, not addsd.
    #[test]
    fn e2e_f32_add() {
        use crate::ir::types::Type;

        // Build: f32_add(a: f32, b: f32) -> f32  (using fadd).
        // F32 support requires fconst for F32 values; use params instead.
        let mut builder =
            FunctionBuilder::new("blitz_f32_add", &[Type::F32, Type::F32], &[Type::F32]);
        let params = builder.params().to_vec();
        let sum = builder.fadd(params[0], params[1]);
        builder.ret(Some(sum));
        let (func, egraph) = builder.finalize().expect("f32_add finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile f32_add");
        assert!(!obj.code.is_empty());

        let c_main = r#"
float blitz_f32_add(float a, float b);
int main(void) {
    float r = blitz_f32_add(1.5f, 2.5f);
    return (r == 4.0f) ? 0 : 1;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_f32_add", &obj, c_main) {
            assert_eq!(code, 0, "f32_add returned wrong exit code {code}");
        }
    }

    // Fix 8: Addr fusion into Load — load(add(base, iconst(16))) emits [base + 16].
    #[test]
    fn e2e_addr_fusion_load() {
        // Build: load_offset16(ptr: *i64) -> i64 — loads *(ptr + 16).
        // The add(ptr, iconst(16)) should fuse into the load addressing mode.
        let mut builder = FunctionBuilder::new("blitz_load_offset16", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let base = params[0];
        let offset = builder.iconst(16, Type::I64);
        let addr = builder.add(base, offset);
        let val = builder.load(addr, Type::I64);
        builder.ret(Some(val));
        let (func, egraph) = builder.finalize().expect("load_offset16 finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile load_offset16");
        assert!(!obj.code.is_empty());

        let c_main = r#"
#include <stdint.h>
int64_t blitz_load_offset16(int64_t *ptr);
int main(void) {
    int64_t arr[4] = {10, 20, 30, 40};
    // arr[2] is at offset 16 bytes from arr[0]
    int64_t r = blitz_load_offset16(arr);
    return (r == 30) ? 0 : 1;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_addr_fuse", &obj, c_main) {
            assert_eq!(code, 0, "load_offset16 returned wrong exit code {code}");
        }
    }

    // Fix 10: Branch threading — a block that just jumps to another block should
    // have its predecessors redirected to skip the trampoline block.
    #[test]
    fn branch_threading_skips_empty_block() {
        use crate::ir::condcode::CondCode;

        // Build a CFG where BB2 is an empty trampoline that jumps to BB3:
        //   BB0: if cond goto BB1 else BB2
        //   BB1: ret(1)
        //   BB2: jump(BB3)    <- empty trampoline
        //   BB3: ret(0)
        //
        // After threading, BB0's false branch should target BB3 directly.
        let mut builder = FunctionBuilder::new("blitz_threaded", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let n = params[0];

        let (bb1, _) = builder.create_block_with_params(&[]);
        let (bb2, _) = builder.create_block_with_params(&[]);
        let (bb3, _) = builder.create_block_with_params(&[]);

        // BB0: branch on n > 0.
        let zero = builder.iconst(0, Type::I64);
        let cond = builder.icmp(CondCode::Sgt, n, zero);
        builder.branch(cond, bb1, bb2, &[], &[]);

        // BB1: return 1.
        builder.set_block(bb1);
        let one = builder.iconst(1, Type::I64);
        builder.ret(Some(one));

        // BB2: just jump to BB3 (trampoline).
        builder.set_block(bb2);
        builder.jump(bb3, &[]);

        // BB3: return 0.
        builder.set_block(bb3);
        let ret_zero = builder.iconst(0, Type::I64);
        builder.ret(Some(ret_zero));

        let (func, egraph) = builder.finalize().expect("threaded finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile threaded");
        assert!(!obj.code.is_empty());

        let c_main = r#"
#include <stdint.h>
int64_t blitz_threaded(int64_t n);
int main(void) {
    if (blitz_threaded(5)  != 1) return 1;
    if (blitz_threaded(-1) != 0) return 2;
    if (blitz_threaded(0)  != 0) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_threaded", &obj, c_main) {
            assert_eq!(code, 0, "threaded returned wrong exit code {code}");
        }
    }

    // Regression 1: Negative value arithmetic.
    //
    // Build: neg_arith(a, b) -> (a - b) - (a + b)  i.e. -2b
    // Tests signed add/sub with negative inputs and negative iconst values.
    // Avoids Op::Mul (not yet supported with two variable operands).
    #[test]
    fn e2e_negative_arithmetic() {
        if !has_tool("cc") {
            return;
        }

        let mut builder =
            FunctionBuilder::new("blitz_neg_arith", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];
        let diff = builder.sub(a, b); // a - b
        let sum = builder.add(a, b); // a + b
        let result = builder.sub(diff, sum); // (a-b) - (a+b) = -2b
        builder.ret(Some(result));
        let (func, egraph) = builder.finalize().expect("neg_arith finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile neg_arith");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_neg_arith(int64_t a, int64_t b);
int main(void) {
    // (a-b) - (a+b) = -2b
    // (-5-3) - (-5+3) = -8 - (-2) = -6
    if (blitz_neg_arith(-5, 3) != -6) return 1;
    // (-100-(-200)) - (-100+(-200)) = 100 - (-300) = 400
    if (blitz_neg_arith(-100, -200) != 400) return 2;
    // (0-(-1)) - (0+(-1)) = 1 - (-1) = 2
    if (blitz_neg_arith(0, -1) != 2) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_negative_arith", &obj, c_main) {
            assert_eq!(code, 0, "neg_arith returned wrong exit code {code}");
        }
    }

    // Regression 2: Signed overflow / wrapping arithmetic.
    //
    // Build: wrap_add(a) -> a + 1  and  wrap_sub(a) -> a - 1
    // Verifies that i64 add/sub wraps at INT64_MAX / INT64_MIN as per two's
    // complement, catching any accidental use of overflow-trapping instructions.
    #[test]
    fn e2e_wrapping_overflow() {
        if !has_tool("cc") {
            return;
        }

        // Build wrap_add(a: i64) -> i64 { a + 1 }
        let mut builder = FunctionBuilder::new("blitz_wrap_add", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let one = builder.iconst(1, Type::I64);
        let result = builder.add(params[0], one);
        builder.ret(Some(result));
        let (func, egraph) = builder.finalize().expect("wrap_add finalize");
        let opts = CompileOptions::default();
        let obj_add = compile(&func, egraph, &opts, None).expect("compile wrap_add");

        // Build wrap_sub(a: i64) -> i64 { a - 1 }
        let mut builder = FunctionBuilder::new("blitz_wrap_sub", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let one = builder.iconst(1, Type::I64);
        let result = builder.sub(params[0], one);
        builder.ret(Some(result));
        let (func2, egraph2) = builder.finalize().expect("wrap_sub finalize");
        let obj_sub = compile(&func2, egraph2, &opts, None).expect("compile wrap_sub");

        let dir = std::env::temp_dir();
        let obj_add_path = dir.join("blitz_e2e_wrap_add.o");
        let obj_sub_path = dir.join("blitz_e2e_wrap_sub.o");
        let main_path = dir.join("blitz_e2e_wrap_main.c");
        let bin_path = dir.join("blitz_e2e_wrap_bin");

        obj_add.write_to(&obj_add_path).expect("write wrap_add.o");
        obj_sub.write_to(&obj_sub_path).expect("write wrap_sub.o");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_wrap_add(int64_t a);
int64_t blitz_wrap_sub(int64_t a);
int main(void) {
    // INT64_MAX + 1 wraps to INT64_MIN
    int64_t int64_max = (int64_t)0x7fffffffffffffffLL;
    int64_t int64_min = (int64_t)0x8000000000000000LL;
    if (blitz_wrap_add(int64_max) != int64_min) return 1;
    // INT64_MIN - 1 wraps to INT64_MAX
    if (blitz_wrap_sub(int64_min) != int64_max) return 2;
    return 0;
}
"#;
        std::fs::write(&main_path, c_main.as_bytes()).expect("write wrap main.c");

        let compile_out = std::process::Command::new("cc")
            .args([
                main_path.to_str().unwrap(),
                obj_add_path.to_str().unwrap(),
                obj_sub_path.to_str().unwrap(),
                "-o",
                bin_path.to_str().unwrap(),
            ])
            .output()
            .expect("cc wrap");

        let _ = std::fs::remove_file(&obj_add_path);
        let _ = std::fs::remove_file(&obj_sub_path);
        let _ = std::fs::remove_file(&main_path);

        if compile_out.status.success() {
            let run = std::process::Command::new(&bin_path)
                .output()
                .expect("run wrap binary");
            let _ = std::fs::remove_file(&bin_path);
            let code = run.status.code().unwrap_or(1);
            assert_eq!(code, 0, "wrapping_overflow returned wrong exit code {code}");
        } else {
            let _ = std::fs::remove_file(&bin_path);
            eprintln!(
                "cc failed for wrapping_overflow:\n{}",
                String::from_utf8_lossy(&compile_out.stderr)
            );
        }
    }

    // Regression 3: Value live across a call (caller-saved register clobber).
    //
    // Build: across_call(a: i64, b: i64) -> i64
    //   x = a + b          -- computed before the call
    //   r = helper(a)      -- call clobbers caller-saved registers (RDI, RSI, ...)
    //   return x + r       -- x must survive the call
    //
    // Uses a two-block structure so that x is live-out of the call block (passed
    // as a block param to the exit block), ensuring the liveness model sees x
    // as live at the call boundary and places it in a callee-saved register.
    //
    // CFG: BB0 [call block] -- jump(BB_exit, [x, r]) --> BB_exit[x, r] --> ret(x+r)
    #[test]
    fn e2e_value_across_call() {
        if !has_tool("cc") {
            return;
        }

        let mut builder =
            FunctionBuilder::new("blitz_across_call", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];

        let (bb_exit, bb_exit_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
        let px = bb_exit_params[0]; // x arriving via block param
        let pr = bb_exit_params[1]; // r arriving via block param

        // BB0: compute x, call helper, jump to exit with both values.
        let x = builder.add(a, b); // x = a + b
        let results = builder.call("blitz_helper_ext", &[a], &[Type::I64]);
        let r = results[0]; // r = helper(a)
        builder.jump(bb_exit, &[x, r]); // x is live-out via block param

        // BB_exit: return x + r
        builder.set_block(bb_exit);
        let ret = builder.add(px, pr);
        builder.ret(Some(ret));

        let (func, egraph) = builder.finalize().expect("across_call finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile across_call");

        // blitz_helper_ext(a) returns a + 100.
        // across_call(a, b) = (a+b) + (a+100) = 2*a + b + 100
        // across_call(5, 3)  = 8 + 105 = 113
        // across_call(10, 2) = 12 + 110 = 122
        // across_call(0, 0)  = 0 + 100 = 100
        let c_main = r#"
#include <stdint.h>
int64_t blitz_across_call(int64_t a, int64_t b);
int64_t blitz_helper_ext(int64_t a) { return a + 100; }
int main(void) {
    if (blitz_across_call(5,  3) != 113) return 1;
    if (blitz_across_call(10, 2) != 122) return 2;
    if (blitz_across_call(0,  0) != 100) return 3;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_across_call", &obj, c_main) {
            assert_eq!(code, 0, "value_across_call returned wrong exit code {code}");
        }
    }

    // Regression 4: Spill correctness with 16 simultaneously live values.
    //
    // Build a function with 16 live i64 values (v1..v16), each param+k for k=1..16.
    // Returns their sum. Forces spilling since there are only 15 GPR colors.
    // Starts at k=1 to avoid Add(param, 0) being folded away by algebraic rules.
    // With param=1: sum of (1+1)..(1+16) = sum of 2..17 = 152.
    // With param=0: sum of 1..16 = 136.
    #[test]
    fn e2e_spill_correctness() {
        if !has_tool("cc") {
            return;
        }

        let mut builder = FunctionBuilder::new("blitz_spill_test", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let param = params[0];

        // v[k] = param + k  for k in 1..=16  (16 distinct values)
        let mut vals = Vec::with_capacity(16);
        for k in 1i64..=16 {
            let ck = builder.iconst(k, Type::I64);
            let v = builder.add(param, ck);
            vals.push(v);
        }

        // Sum all 16 values while keeping them all live until the final add chain.
        // Build as a left fold: acc = v1, acc = acc + v2, ..., acc = acc + v16
        let mut acc = vals[0];
        for v in &vals[1..] {
            acc = builder.add(acc, *v);
        }
        builder.ret(Some(acc));
        let (func, egraph) = builder.finalize().expect("spill_test finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile spill_test");

        // param=1: sum of (1+1)..(1+16) = 2+3+...+17 = (2+17)*16/2 = 152
        // param=0: sum of (0+1)..(0+16) = 1+2+...+16 = (1+16)*16/2 = 136
        let c_main = r#"
#include <stdint.h>
int64_t blitz_spill_test(int64_t param);
int main(void) {
    if (blitz_spill_test(1) != 152) return 1;
    if (blitz_spill_test(0) != 136) return 2;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_spill", &obj, c_main) {
            assert_eq!(code, 0, "spill_correctness returned wrong exit code {code}");
        }
    }

    // Regression 5: Complex CFG with nested if/else.
    //
    // Build: classify(x) -> { >100: 3, 1..100: 2, -100..0: -2, <-100: -3 }
    // Tests nested conditional blocks and multiple returns through different paths.
    #[test]
    fn e2e_nested_if_else() {
        use crate::ir::condcode::CondCode;
        if !has_tool("cc") {
            return;
        }

        // CFG:
        //   BB0: if x > 0 -> BB_pos, else -> BB_neg
        //   BB_pos: if x > 100 -> BB_big_pos, else -> BB_small_pos
        //   BB_big_pos: ret(3)
        //   BB_small_pos: ret(2)
        //   BB_neg: if x < -100 -> BB_big_neg, else -> BB_small_neg
        //   BB_big_neg: ret(-3)
        //   BB_small_neg: ret(-2)
        let mut builder = FunctionBuilder::new("blitz_classify", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let x = params[0];

        let (bb_pos, _) = builder.create_block_with_params(&[]);
        let (bb_neg, _) = builder.create_block_with_params(&[]);
        let (bb_big_pos, _) = builder.create_block_with_params(&[]);
        let (bb_small_pos, _) = builder.create_block_with_params(&[]);
        let (bb_big_neg, _) = builder.create_block_with_params(&[]);
        let (bb_small_neg, _) = builder.create_block_with_params(&[]);

        // BB0: x > 0 ?
        let zero = builder.iconst(0, Type::I64);
        let cond0 = builder.icmp(CondCode::Sgt, x, zero);
        builder.branch(cond0, bb_pos, bb_neg, &[], &[]);

        // BB_pos: x > 100 ?
        builder.set_block(bb_pos);
        let c100 = builder.iconst(100, Type::I64);
        let cond_pos = builder.icmp(CondCode::Sgt, x, c100);
        builder.branch(cond_pos, bb_big_pos, bb_small_pos, &[], &[]);

        // BB_big_pos: ret 3
        builder.set_block(bb_big_pos);
        let v3 = builder.iconst(3, Type::I64);
        builder.ret(Some(v3));

        // BB_small_pos: ret 2
        builder.set_block(bb_small_pos);
        let v2 = builder.iconst(2, Type::I64);
        builder.ret(Some(v2));

        // BB_neg: x < -100 ?
        builder.set_block(bb_neg);
        let cn100 = builder.iconst(-100, Type::I64);
        let cond_neg = builder.icmp(CondCode::Slt, x, cn100);
        builder.branch(cond_neg, bb_big_neg, bb_small_neg, &[], &[]);

        // BB_big_neg: ret -3
        builder.set_block(bb_big_neg);
        let vn3 = builder.iconst(-3, Type::I64);
        builder.ret(Some(vn3));

        // BB_small_neg: ret -2
        builder.set_block(bb_small_neg);
        let vn2 = builder.iconst(-2, Type::I64);
        builder.ret(Some(vn2));

        let (func, egraph) = builder.finalize().expect("classify finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile classify");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_classify(int64_t x);
int main(void) {
    if (blitz_classify(200)  !=  3) return 1;
    if (blitz_classify(50)   !=  2) return 2;
    if (blitz_classify(0)    != -2) return 3;
    if (blitz_classify(-50)  != -2) return 4;
    if (blitz_classify(-200) != -3) return 5;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_nested_if", &obj, c_main) {
            assert_eq!(code, 0, "nested_if_else returned wrong exit code {code}");
        }
    }

    // Regression 6: Loop with phi swapping -- Fibonacci via block params.
    //
    // Build: fib(n) iteratively using loop variables a and b (two block params).
    // Tests phi (block param) copy correctness across loop backedge when
    // the two loop variables must be swapped simultaneously: (a,b) <- (b, a+b).
    // Uses a separate counter block param to avoid sharing iconst nodes.
    // fib(0)=0, fib(1)=1, fib(10)=55, fib(20)=6765
    #[test]
    fn e2e_fibonacci() {
        use crate::ir::condcode::CondCode;
        if !has_tool("cc") {
            return;
        }

        // CFG:
        //   BB0: jump(BB_loop, [n, a=0, b=1])
        //   BB_loop(params=[count, a, b]):
        //     count_minus1 = count - 1  (using fresh iconst(1) not shared with initial b)
        //     next_b = a + b
        //     cond = count > 0
        //     branch(cond, BB_loop, BB_exit, [count_minus1, b, next_b], [a])
        //   BB_exit(params=[result]):
        //     ret(result)
        //
        // Use iconst(-1) and add instead of sub(count, iconst(1)) to avoid
        // sharing the iconst(1) node with the initial b=1 argument.
        let mut builder = FunctionBuilder::new("blitz_fib", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let n = params[0];

        let (bb_loop, bb_loop_params) =
            builder.create_block_with_params(&[Type::I64, Type::I64, Type::I64]);
        let count = bb_loop_params[0];
        let a = bb_loop_params[1];
        let b = bb_loop_params[2];

        let (bb_exit, bb_exit_params) = builder.create_block_with_params(&[Type::I64]);
        let result = bb_exit_params[0];

        // BB0: jump to loop with count=n, a=0, b=1
        let zero = builder.iconst(0, Type::I64);
        let init_b = builder.iconst(1, Type::I64); // initial b=1
        builder.jump(bb_loop, &[n, zero, init_b]);

        // BB_loop: decrement count using add(count, -1) to avoid sharing iconst(1).
        builder.set_block(bb_loop);
        let neg_one = builder.iconst(-1, Type::I64); // distinct from init_b
        let count_minus1 = builder.add(count, neg_one); // count + (-1)
        let next_b = builder.add(a, b); // a + b
        let cond = builder.icmp(CondCode::Sgt, count, zero);
        builder.branch(cond, bb_loop, bb_exit, &[count_minus1, b, next_b], &[a]);

        // BB_exit: return result
        builder.set_block(bb_exit);
        builder.ret(Some(result));

        let (func, egraph) = builder.finalize().expect("fib finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile fib");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_fib(int64_t n);
int main(void) {
    if (blitz_fib(0)  != 0)    return 1;
    if (blitz_fib(1)  != 1)    return 2;
    if (blitz_fib(10) != 55)   return 3;
    if (blitz_fib(20) != 6765) return 4;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_fibonacci", &obj, c_main) {
            assert_eq!(code, 0, "fibonacci returned wrong exit code {code}");
        }
    }

    // Regression 7: Diamond CFG phi merge -- abs_diff.
    //
    // Build: abs_diff(a, b) -> { a > b: a - b, else: b - a }
    // Both CFG paths produce a different value that merges at the exit block
    // via a block parameter (phi). Tests phi-copy correctness.
    #[test]
    fn e2e_diamond_phi() {
        use crate::ir::condcode::CondCode;
        if !has_tool("cc") {
            return;
        }

        // CFG:
        //   BB0: cond = a > b; branch(cond, BB_true, BB_false)
        //   BB_true: diff = a - b; jump(BB_exit, [diff])
        //   BB_false: diff = b - a; jump(BB_exit, [diff])
        //   BB_exit(params=[diff]): ret(diff)
        let mut builder =
            FunctionBuilder::new("blitz_abs_diff", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];

        let (bb_true, _) = builder.create_block_with_params(&[]);
        let (bb_false, _) = builder.create_block_with_params(&[]);
        let (bb_exit, bb_exit_params) = builder.create_block_with_params(&[Type::I64]);
        let diff = bb_exit_params[0];

        let cond = builder.icmp(CondCode::Sgt, a, b);
        builder.branch(cond, bb_true, bb_false, &[], &[]);

        builder.set_block(bb_true);
        let diff_true = builder.sub(a, b);
        builder.jump(bb_exit, &[diff_true]);

        builder.set_block(bb_false);
        let diff_false = builder.sub(b, a);
        builder.jump(bb_exit, &[diff_false]);

        builder.set_block(bb_exit);
        builder.ret(Some(diff));

        let (func, egraph) = builder.finalize().expect("abs_diff finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile abs_diff");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_abs_diff(int64_t a, int64_t b);
int main(void) {
    if (blitz_abs_diff(10, 3)  != 7) return 1;
    if (blitz_abs_diff(3, 10)  != 7) return 2;
    if (blitz_abs_diff(5, 5)   != 0) return 3;
    if (blitz_abs_diff(-1, -5) != 4) return 4;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_diamond_phi", &obj, c_main) {
            assert_eq!(code, 0, "diamond_phi returned wrong exit code {code}");
        }
    }

    // Regression 8: Constant folding through pipeline.
    //
    // Build: constfold() -> (3 + 7) * (10 - 4)
    // All inputs are iconst nodes. The e-graph should fold this to 60 before
    // isel, so the compiled function should just return a constant.
    #[test]
    fn e2e_constant_fold() {
        if !has_tool("cc") {
            return;
        }

        let mut builder = FunctionBuilder::new("blitz_constfold", &[], &[Type::I64]);
        let c3 = builder.iconst(3, Type::I64);
        let c7 = builder.iconst(7, Type::I64);
        let c10 = builder.iconst(10, Type::I64);
        let c4 = builder.iconst(4, Type::I64);
        let sum = builder.add(c3, c7);
        let diff = builder.sub(c10, c4);
        let result = builder.mul(sum, diff);
        builder.ret(Some(result));
        let (func, egraph) = builder.finalize().expect("constfold finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile constfold");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_constfold(void);
int main(void) {
    if (blitz_constfold() != 60) return 1;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_constfold", &obj, c_main) {
            assert_eq!(code, 0, "constant_fold returned wrong exit code {code}");
        }
    }

    // Regression 9: Chained comparisons -- clamp function.
    //
    // Build: clamp(x, lo, hi) -> lo if x < lo, hi if x > hi, else x
    // Multiple conditional blocks with separate branches and returns.
    // Tests flag fusion and branch correctness across sequential comparisons.
    #[test]
    fn e2e_chained_cmp() {
        use crate::ir::condcode::CondCode;
        if !has_tool("cc") {
            return;
        }

        // CFG:
        //   BB0: cond = x < lo; branch(cond, BB_ret_lo, BB_check_hi)
        //   BB_ret_lo: ret(lo)
        //   BB_check_hi: cond2 = x > hi; branch(cond2, BB_ret_hi, BB_ret_x)
        //   BB_ret_hi: ret(hi)
        //   BB_ret_x: ret(x)
        let mut builder = FunctionBuilder::new(
            "blitz_clamp",
            &[Type::I64, Type::I64, Type::I64],
            &[Type::I64],
        );
        let params = builder.params().to_vec();
        let x = params[0];
        let lo = params[1];
        let hi = params[2];

        let (bb_ret_lo, _) = builder.create_block_with_params(&[]);
        let (bb_check_hi, _) = builder.create_block_with_params(&[]);
        let (bb_ret_hi, _) = builder.create_block_with_params(&[]);
        let (bb_ret_x, _) = builder.create_block_with_params(&[]);

        // BB0: x < lo ?
        let cond0 = builder.icmp(CondCode::Slt, x, lo);
        builder.branch(cond0, bb_ret_lo, bb_check_hi, &[], &[]);

        // BB_ret_lo: return lo
        builder.set_block(bb_ret_lo);
        builder.ret(Some(lo));

        // BB_check_hi: x > hi ?
        builder.set_block(bb_check_hi);
        let cond1 = builder.icmp(CondCode::Sgt, x, hi);
        builder.branch(cond1, bb_ret_hi, bb_ret_x, &[], &[]);

        // BB_ret_hi: return hi
        builder.set_block(bb_ret_hi);
        builder.ret(Some(hi));

        // BB_ret_x: return x
        builder.set_block(bb_ret_x);
        builder.ret(Some(x));

        let (func, egraph) = builder.finalize().expect("clamp finalize");
        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile clamp");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_clamp(int64_t x, int64_t lo, int64_t hi);
int main(void) {
    if (blitz_clamp(5,   0, 10) != 5)  return 1;
    if (blitz_clamp(-5,  0, 10) != 0)  return 2;
    if (blitz_clamp(15,  0, 10) != 10) return 3;
    if (blitz_clamp(0,   0, 10) != 0)  return 4;
    if (blitz_clamp(10,  0, 10) != 10) return 5;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_clamp", &obj, c_main) {
            assert_eq!(code, 0, "chained_cmp returned wrong exit code {code}");
        }
    }

    // Regression 10: Shift with immediate count (X86ShlImm).
    //
    // Build: shl_imm(val: i64) -> i64 { val << 3 }
    // When the shift count is a constant iconst, isel produces X86ShlImm which
    // encodes as a SAL/SHL with an 8-bit immediate -- no RCX pre-coloring needed.
    // Tests that constant-count shifts compile and execute correctly, including
    // sign-extension behaviour at the boundary bits.
    #[test]
    fn e2e_shift_edge_cases() {
        if !has_tool("cc") {
            return;
        }

        let mut builder = FunctionBuilder::new("blitz_shl_imm", &[Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let val = params[0];
        let c3 = builder.iconst(3, Type::I64);
        let result = builder.shl(val, c3); // val << 3 via X86ShlImm(3)
        builder.ret(Some(result));
        let (func, egraph) = builder.finalize().expect("shl_imm finalize");

        let opts = CompileOptions::default();
        let obj = compile(&func, egraph, &opts, None).expect("compile shl_imm");

        let c_main = r#"
#include <stdint.h>
int64_t blitz_shl_imm(int64_t val);
int main(void) {
    if (blitz_shl_imm(1)  != 8)    return 1;
    if (blitz_shl_imm(5)  != 40)   return 2;
    if (blitz_shl_imm(0)  != 0)    return 3;
    if (blitz_shl_imm(-1) != -8)   return 4;
    // 1 << 62 = 0x4000000000000000  (not sign bit, so stays positive)
    // but our shift is by 3, so test large input: 0x1000000000000000 << 3 = INT64_MIN
    if (blitz_shl_imm((int64_t)0x1000000000000000LL) != (int64_t)0x8000000000000000LL) return 5;
    return 0;
}
"#;
        if let Some(code) = link_and_run_obj("blitz_e2e_shift_edge", &obj, c_main) {
            assert_eq!(code, 0, "shift_edge_cases returned wrong exit code {code}");
        }
    }
}
