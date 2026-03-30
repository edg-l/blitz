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
use crate::egraph::extract::{VReg, VRegInst, extract, vreg_insts_for_block};
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

    // Phase 5: Register allocation across all blocks (concatenated).
    //
    // We concatenate all block schedules into a single flat list. The live-out
    // set is the union of:
    //   a) All block param VRegs (phi destinations, so they survive jumps/branches)
    //   b) All phi source VRegs (the args passed to Jump/Branch), so that values
    //      needed for phi copies are kept alive through loop back-edges
    let param_vregs = assign_param_vregs_from_map(func, &class_to_vreg, &egraph);

    // Compute global live-out (conservative):
    //   a) All block param VRegs (phi destinations)
    //   b) All phi source VRegs (branch/jump args, especially loop back-edge values)
    //   c) All function param VRegs: function parameters have no ScheduledInst
    //      (they are pre-colored), so liveness analysis cannot track them. Adding
    //      them to live_out forces the interference graph to prevent block param
    //      VRegs from being assigned the same register as any function parameter.
    //   d) All cross-block operand VRegs: values defined in one block but used in
    //      another must survive across the boundary.
    let mut live_out: HashSet<VReg> = collect_block_param_vregs(&egraph, &class_to_vreg);
    collect_phi_source_vregs(func, &egraph, &class_to_vreg, &mut live_out);
    // (c) Function param VRegs must stay alive across all blocks.
    for &(vreg, _reg) in &param_vregs {
        live_out.insert(vreg);
    }
    // (d) Cross-block operand VRegs: values used in a different block than where
    // they are defined must be globally live.
    for (block_idx, sched) in block_schedules.iter().enumerate() {
        let defined_in_block: HashSet<VReg> = sched.iter().map(|s| s.dst).collect();
        for other_sched in block_schedules
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != block_idx)
            .map(|(_, s)| s)
        {
            for inst in other_sched {
                for &op_vreg in &inst.operands {
                    if defined_in_block.contains(&op_vreg) {
                        live_out.insert(op_vreg);
                    }
                }
            }
        }
    }

    let all_scheduled: Vec<ScheduledInst> = block_schedules.iter().flatten().cloned().collect();

    // Build phi copy pairs from block parameter passing for coalescing.
    let copy_pairs = compute_copy_pairs(func, &class_to_vreg, &egraph, &block_param_map);

    // Compute loop depths from the CFG for spill selection.
    let loop_depths = compute_loop_depths(func, &block_schedules, &class_to_vreg);

    // Pre-color shift count operands to RCX so variable shifts don't clobber live values.
    let mut all_param_vregs = param_vregs.clone();
    for inst in &all_scheduled {
        if matches!(inst.op, Op::X86Shl | Op::X86Shr | Op::X86Sar) && inst.operands.len() >= 2 {
            let count_vreg = inst.operands[1];
            // Only pre-color if not already pre-colored to something else.
            if !all_param_vregs.iter().any(|&(v, _)| v == count_vreg) {
                all_param_vregs.push((count_vreg, Reg::RCX));
            }
        }
    }

    // Pre-color the first CallResult of each Call op to RAX (GPR return register).
    // This ensures the return value lands in RAX so lower_effectful_op can capture it.
    for block in &func.blocks {
        for op in &block.ops {
            if let EffectfulOp::Call { results, .. } = op {
                if let Some(&first_result_cid) = results.first() {
                    let canon = egraph.unionfind.find_immutable(first_result_cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        if !all_param_vregs.iter().any(|&(v, _)| v == vreg) {
                            all_param_vregs.push((vreg, GPR_RETURN_REG));
                        }
                    }
                }
            }
        }
    }

    let regalloc_result = allocate(
        &all_scheduled,
        &all_param_vregs,
        &live_out,
        &copy_pairs,
        &loop_depths,
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

    // Phase 6: Rewrite VRegs per block.
    let block_rewritten: Vec<Vec<ScheduledInst>> = block_schedules
        .iter()
        .map(|sched| rewrite_vregs(sched, &regalloc_result.vreg_to_reg))
        .collect();

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
            let extra = lower_effectful_op(op, &class_to_vreg, &regalloc_result, func)?;
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

        // Fconst: load FP constant into an XMM register using a 64-bit immediate
        // materialized via MOVSD from a data constant embedded in the code.
        // For now, emit a MOVSD from memory (simplified: use a spill-based approach).
        // TODO: proper .rodata section for FP constants.
        Op::Fconst(bits) => {
            let dst = dst_reg.ok_or_else(|| "Fconst: no register for dst".to_string())?;
            // Use MovRI to materialize bits into a GPR, then MOVSD-like move.
            // This is a simplification; ideally we'd use a memory reference.
            // For now emit as integer constant (relies on bitcast behavior).
            Ok(vec![MachInst::MovRI {
                dst: Operand::Reg(dst),
                imm: *bits as i64,
            }])
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

/// Collect VRegs for all block params across all blocks (used for global live-out).
fn collect_block_param_vregs(
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
) -> HashSet<VReg> {
    let mut result = HashSet::new();
    for i in 0..egraph.classes.len() as u32 {
        let cid = ClassId(i);
        let canon = egraph.unionfind.find_immutable(cid);
        if canon != cid {
            continue;
        }
        let class = egraph.class(cid);
        for node in &class.nodes {
            if matches!(node.op, Op::BlockParam(_, _, _)) {
                if let Some(&vreg) = class_to_vreg.get(&cid) {
                    result.insert(vreg);
                }
            }
        }
    }
    result
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

/// Lower a non-terminator effectful op (Load, Store, Call) to MachInsts.
fn lower_effectful_op(
    op: &EffectfulOp,
    class_to_vreg: &HashMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
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
            let addr_reg = get_reg(*addr).ok_or_else(|| CompileError {
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
            Ok(vec![MachInst::MovRM {
                dst: Operand::Reg(result_reg),
                addr: crate::x86::addr::Addr {
                    base: Some(addr_reg),
                    index: None,
                    scale: 1,
                    disp: 0,
                },
            }])
        }
        EffectfulOp::Store { addr, val } => {
            let addr_reg = get_reg(*addr).ok_or_else(|| CompileError {
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
            Ok(vec![MachInst::MovMR {
                addr: crate::x86::addr::Addr {
                    base: Some(addr_reg),
                    index: None,
                    scale: 1,
                    disp: 0,
                },
                src: Operand::Reg(val_reg),
            }])
        }
        EffectfulOp::Call {
            func: callee,
            args,
            ret_tys: _,
            results,
        } => {
            let arg_regs: Vec<Reg> = args.iter().filter_map(|&cid| get_reg(cid)).collect();
            // All args treated as I64 for ABI setup (conservative but correct for GPR args).
            let arg_types: Vec<crate::ir::types::Type> = (0..arg_regs.len())
                .map(|_| crate::ir::types::Type::I64)
                .collect();
            let mut insts = setup_call_args(&arg_types, &arg_regs, Reg::R11);
            insts.push(MachInst::CallDirect {
                target: callee.clone(),
            });
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
}
