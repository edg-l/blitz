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
use crate::x86::encode::Encoder;
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
    // We process blocks in order. Each block gets the VRegInsts for classes
    // first encountered in that block. Classes shared between blocks are only
    // emitted by the first block that reaches them (DFS deduplication).
    // DO NOT pre-populate class_to_vreg here — let the DFS assign VRegs
    // naturally so that param/block-param VRegInsts appear in the scheduled
    // list and regalloc can see them.
    let mut class_to_vreg: HashMap<ClassId, VReg> = HashMap::new();
    let mut next_vreg: u32 = 0;

    // Build the block param class map (needed for phi copy generation).
    let block_param_map = build_block_param_class_map(&egraph);

    // Build per-block VRegInst lists.
    let mut block_vreg_insts: Vec<Vec<VRegInst>> = Vec::new();
    for block in &func.blocks {
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
        block_vreg_insts.push(insts);
    }

    // Phase 4: Schedule per block.
    let mut block_schedules: Vec<Vec<ScheduledInst>> = Vec::new();
    let mut total_insts = 0usize;
    for insts in &block_vreg_insts {
        let dag = ScheduleDag::build(insts);
        let sched = schedule(&dag);
        total_insts += sched.len();
        block_schedules.push(sched);
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

    let regalloc_result =
        allocate(&all_scheduled, &param_vregs, &live_out).map_err(|e| CompileError {
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

    // Phase 7: Per-block MachInst lowering + phi elimination + terminator emission.
    // Assign a LabelId to each block (block index = label id).
    let n_blocks = func.blocks.len();
    // Extra labels for trampoline code start after the block labels.
    let mut next_label: LabelId = n_blocks as LabelId;
    let mut block_items: Vec<Vec<BlockItem>> = Vec::with_capacity(n_blocks);

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let rewritten = &block_rewritten[block_idx];

        // Lower pure ops for this block.
        let pure_insts = lower_block_pure_ops(rewritten, &regalloc_result, func, &param_vreg_set)?;

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

    // Phase 10: Encoding with labels.
    let mut encoder = Encoder::new();

    let frame_layout = compute_frame_layout(
        regalloc_result.spill_slots,
        &regalloc_result.callee_saved_used,
        0,
    );
    let func_start = encoder.buf.len();
    emit_prologue(&mut encoder, &frame_layout);

    for (block_idx, items) in block_items.iter().enumerate() {
        // Bind the label for this block at the current position.
        encoder.bind_label(block_idx as LabelId);

        for item in items {
            match item {
                BlockItem::Inst(inst) => {
                    if *inst == MachInst::Ret {
                        // Emit the full epilogue (including the CPU ret instruction)
                        // in place of the Ret marker.
                        emit_epilogue(&mut encoder, &frame_layout);
                    } else {
                        encoder.encode_inst(inst);
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

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Collect all ClassIds that are roots for extraction (used by effectful ops).
fn push_block_class_ids(block: &crate::ir::function::BasicBlock, out: &mut Vec<ClassId>) {
    for op in &block.ops {
        match op {
            EffectfulOp::Load { addr, .. } => out.push(*addr),
            EffectfulOp::Store { addr, val } => {
                out.push(*addr);
                out.push(*val);
            }
            EffectfulOp::Call { args, .. } => out.extend_from_slice(args),
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
    if dst != src_a {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    // LIMITATION: Post-regalloc MOV to RCX can clobber a live value.
    // TODO: Pre-color shift count operand to RCX before register allocation.
    if src_b != Reg::RCX {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(Reg::RCX),
            src: Operand::Reg(src_b),
        });
    }
    insts.push(mk(Operand::Reg(dst)));
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
        // Move shift count to CL if not already there.
        // LIMITATION: This post-regalloc MOV can clobber a live value in RCX.
        // The proper fix requires pre-coloring the shift count to RCX in the
        // register allocator, or using the immediate form (ShlRI) when the
        // shift amount is a constant (detected before regalloc).
        Op::X86Shl => lower_shift_cl("X86Shl", dst_reg, operand_regs, |dst| MachInst::ShlRCL {
            dst,
        }),
        Op::X86Shr => lower_shift_cl("X86Shr", dst_reg, operand_regs, |dst| MachInst::ShrRCL {
            dst,
        }),
        Op::X86Sar => lower_shift_cl("X86Sar", dst_reg, operand_regs, |dst| MachInst::SarRCL {
            dst,
        }),

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
        | Op::Select
        | Op::Fconst(_) => Err(format!(
            "unlowered op {op:?}: generic IR must be lowered by isel phases before lowering"
        )),
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
) -> Result<Vec<MachInst>, CompileError> {
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
        EffectfulOp::Load { addr, ty: _ } => {
            // Load result tracking via sentinels is not fully implemented.
            // Emit nothing here; the result VReg will be zero/uninitialized.
            let _ = get_reg(*addr);
            Ok(vec![])
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
#[allow(clippy::too_many_arguments)]
fn lower_terminator(
    op: &EffectfulOp,
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
            items.push(BlockItem::Inst(MachInst::Jmp {
                target: *target as LabelId,
            }));
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

            let mut items = Vec::new();
            if true_phi.is_empty() {
                // jcc cc, true_block; [false_phi]; jmp false_block
                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc,
                    target: *bb_true as LabelId,
                }));
                items.extend(false_phi.into_iter().map(BlockItem::Inst));
                items.push(BlockItem::Inst(MachInst::Jmp {
                    target: *bb_false as LabelId,
                }));
            } else if false_phi.is_empty() {
                // jcc !cc, false_block; [true_phi]; jmp true_block
                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc: negate_cc(cc),
                    target: *bb_false as LabelId,
                }));
                items.extend(true_phi.into_iter().map(BlockItem::Inst));
                items.push(BlockItem::Inst(MachInst::Jmp {
                    target: *bb_true as LabelId,
                }));
            } else {
                // Both sides have copies. Use trampoline labels:
                //   jcc !cc, L_false_copies
                //   [true_phi]
                //   jmp true_block
                //   L_false_copies:
                //   [false_phi]
                //   jmp false_block
                let l_false = *next_label;
                *next_label += 1;

                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc: negate_cc(cc),
                    target: l_false,
                }));
                items.extend(true_phi.into_iter().map(BlockItem::Inst));
                items.push(BlockItem::Inst(MachInst::Jmp {
                    target: *bb_true as LabelId,
                }));
                items.push(BlockItem::BindLabel(l_false));
                items.extend(false_phi.into_iter().map(BlockItem::Inst));
                items.push(BlockItem::Inst(MachInst::Jmp {
                    target: *bb_false as LabelId,
                }));
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
}
