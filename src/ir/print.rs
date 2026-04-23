use crate::egraph::extract::ClassVRegMap;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::schedule::scheduler::ScheduledInst;

/// Format a single Op variant as a human-readable string.
pub fn fmt_op(op: &Op) -> String {
    match op {
        // Arithmetic
        Op::Add => "add".into(),
        Op::Sub => "sub".into(),
        Op::Mul => "mul".into(),
        Op::UDiv => "udiv".into(),
        Op::SDiv => "sdiv".into(),
        Op::URem => "urem".into(),
        Op::SRem => "srem".into(),

        // Bitwise
        Op::And => "and".into(),
        Op::Or => "or".into(),
        Op::Xor => "xor".into(),
        Op::Shl => "shl".into(),
        Op::Shr => "shr".into(),
        Op::Sar => "sar".into(),

        // Conversion
        Op::Sext(ty) => format!("sext({ty:?})"),
        Op::Zext(ty) => format!("zext({ty:?})"),
        Op::Trunc(ty) => format!("trunc({ty:?})"),
        Op::Bitcast(ty) => format!("bitcast({ty:?})"),

        // Constants
        Op::Iconst(val, ty) => format!("iconst({val}, {ty:?})"),
        Op::Fconst(bits, ty) => format!("fconst(0x{bits:x}, {ty:?})"),

        // Parameters
        Op::Param(idx, ty) => format!("param({idx}, {ty:?})"),
        Op::BlockParam(bid, pidx, ty) => format!("block_param(b{bid}, {pidx}, {ty:?})"),

        // Comparison
        Op::Icmp(cc) => format!("icmp({cc:?})"),
        Op::Fcmp(cc) => format!("fcmp({cc:?})"),

        // Float/int conversions
        Op::IntToFloat(ty) => format!("int_to_float({ty:?})"),
        Op::FloatToInt(ty) => format!("float_to_int({ty:?})"),
        Op::FloatExt => "float_ext".into(),
        Op::FloatTrunc => "float_trunc".into(),

        // FP ops
        Op::Fadd => "fadd".into(),
        Op::Fsub => "fsub".into(),
        Op::Fmul => "fmul".into(),
        Op::Fdiv => "fdiv".into(),
        Op::Fsqrt => "fsqrt".into(),

        // Select
        Op::Select => "select".into(),

        // Projections
        Op::Proj0 => "proj0".into(),
        Op::Proj1 => "proj1".into(),

        // x86 ALU
        Op::X86Add => "x86_add".into(),
        Op::X86Sub => "x86_sub".into(),
        Op::X86And => "x86_and".into(),
        Op::X86Or => "x86_or".into(),
        Op::X86Xor => "x86_xor".into(),
        Op::X86Shl => "x86_shl".into(),
        Op::X86Sar => "x86_sar".into(),
        Op::X86Shr => "x86_shr".into(),

        // x86 immediate shifts
        Op::X86ShlImm(n) => format!("x86_shl_imm({n})"),
        Op::X86ShrImm(n) => format!("x86_shr_imm({n})"),
        Op::X86SarImm(n) => format!("x86_sar_imm({n})"),

        // x86 flag-only compare with immediate
        Op::X86CmpI { imm, ty } => format!("x86_cmp_imm({imm}, {ty:?})"),

        // x86 LEA
        Op::X86Lea2 => "x86_lea2".into(),
        Op::X86Lea3 { scale } => format!("x86_lea3(scale={scale})"),
        Op::X86Lea4 { scale, disp } => format!("x86_lea4(scale={scale}, disp={disp})"),

        // x86 multiply/divide
        Op::X86Imul3 => "x86_imul3".into(),
        Op::X86Idiv => "x86_idiv".into(),
        Op::X86Div => "x86_div".into(),

        // x86 conditional ops
        Op::X86Cmov(cc) => format!("x86_cmov({cc:?})"),
        Op::X86Setcc(cc) => format!("x86_setcc({cc:?})"),

        // Addressing
        Op::Addr { scale, disp } => format!("addr(scale={scale}, disp={disp})"),

        // Load/Call result placeholders
        Op::LoadResult(id, ty) => format!("load_result({id}, {ty:?})"),
        Op::CallResult(id, ty) => format!("call_result({id}, {ty:?})"),

        // x86 FP double
        Op::X86Addsd => "x86_addsd".into(),
        Op::X86Subsd => "x86_subsd".into(),
        Op::X86Mulsd => "x86_mulsd".into(),
        Op::X86Divsd => "x86_divsd".into(),
        Op::X86Sqrtsd => "x86_sqrtsd".into(),

        // x86 FP single
        Op::X86Addss => "x86_addss".into(),
        Op::X86Subss => "x86_subss".into(),
        Op::X86Mulss => "x86_mulss".into(),
        Op::X86Divss => "x86_divss".into(),
        Op::X86Sqrtss => "x86_sqrtss".into(),

        // x86 FP conversion
        Op::X86Cvtsi2sd => "x86_cvtsi2sd".into(),
        Op::X86Cvtsi2ss => "x86_cvtsi2ss".into(),
        Op::X86Cvttsd2si(ty) => format!("x86_cvttsd2si({ty:?})"),
        Op::X86Cvttss2si(ty) => format!("x86_cvttss2si({ty:?})"),
        Op::X86Cvtsd2ss => "x86_cvtsd2ss".into(),
        Op::X86Cvtss2sd => "x86_cvtss2sd".into(),

        // x86 FP comparison
        Op::X86Ucomisd => "x86_ucomisd".into(),
        Op::X86Ucomiss => "x86_ucomiss".into(),

        // Stack address
        Op::StackAddr(slot) => format!("stack_addr({slot})"),

        // Global address
        Op::GlobalAddr(name) => format!("global_addr(\"{}\")", name),

        // x86 conversion ops
        Op::X86Movsx { from, to } => format!("x86_movsx({from:?} -> {to:?})"),
        Op::X86Movzx { from, to } => format!("x86_movzx({from:?} -> {to:?})"),
        Op::X86Trunc { from, to } => format!("x86_trunc({from:?} -> {to:?})"),
        Op::X86Bitcast { from, to } => format!("x86_bitcast({from:?} -> {to:?})"),

        // Spill ops
        Op::SpillStore(s) => format!("spill_store({s})"),
        Op::SpillLoad(s) => format!("spill_load({s})"),
        Op::XmmSpillStore(s) => format!("xmm_spill_store({s})"),
        Op::XmmSpillLoad(s) => format!("xmm_spill_load({s})"),

        // Barrier pseudo-ops
        Op::StoreBarrier => "store_barrier".into(),
        Op::VoidCallBarrier => "void_call_barrier".into(),
    }
}

/// Data for a group of pure ops followed by an optional effectful op.
pub struct PrintableGroup {
    pub pure_ops: Vec<ScheduledInst>,
    pub barrier: Option<EffectfulOp>,
}

/// Data for a printable block.
pub struct PrintableBlock {
    pub id: u32,
    pub param_types: Vec<Type>,
    pub groups: Vec<PrintableGroup>,
    pub terminator: EffectfulOp,
}

/// Resolve a ClassId to a VReg number string, or `?{cid}` if not found.
fn resolve_cid(
    cid: ClassId,
    class_to_vreg: &ClassVRegMap,
    egraph_uf: &crate::egraph::unionfind::UnionFind,
) -> String {
    let canon = egraph_uf.find_immutable(cid);
    match class_to_vreg.lookup_any(canon) {
        Some(vreg) => format!("v{}", vreg.0),
        None => format!("?{}", cid.0),
    }
}

/// Format an effectful op for printing.
fn fmt_effectful(
    op: &EffectfulOp,
    class_to_vreg: &ClassVRegMap,
    egraph_uf: &crate::egraph::unionfind::UnionFind,
) -> String {
    match op {
        EffectfulOp::Load { addr, ty, result } => {
            let addr_s = resolve_cid(*addr, class_to_vreg, egraph_uf);
            let result_s = resolve_cid(*result, class_to_vreg, egraph_uf);
            format!("load {ty:?} {addr_s} -> {result_s}")
        }
        EffectfulOp::Store { addr, val, ty } => {
            let addr_s = resolve_cid(*addr, class_to_vreg, egraph_uf);
            let val_s = resolve_cid(*val, class_to_vreg, egraph_uf);
            format!("store {ty:?} {addr_s}, {val_s}")
        }
        EffectfulOp::Call {
            func,
            args,
            results,
            ..
        } => {
            let arg_strs: Vec<String> = args
                .iter()
                .map(|a| resolve_cid(*a, class_to_vreg, egraph_uf))
                .collect();
            let result_strs: Vec<String> = results
                .iter()
                .map(|r| resolve_cid(*r, class_to_vreg, egraph_uf))
                .collect();
            format!(
                "call {}({}) -> [{}]",
                func,
                arg_strs.join(", "),
                result_strs.join(", ")
            )
        }
        EffectfulOp::Branch {
            cond,
            cc,
            bb_true,
            bb_false,
            true_args,
            false_args,
        } => {
            let cond_s = resolve_cid(*cond, class_to_vreg, egraph_uf);
            let true_arg_strs: Vec<String> = true_args
                .iter()
                .map(|a| resolve_cid(*a, class_to_vreg, egraph_uf))
                .collect();
            let false_arg_strs: Vec<String> = false_args
                .iter()
                .map(|a| resolve_cid(*a, class_to_vreg, egraph_uf))
                .collect();
            format!(
                "branch {cc:?} {cond_s} block{bb_true}({}) block{bb_false}({})",
                true_arg_strs.join(", "),
                false_arg_strs.join(", ")
            )
        }
        EffectfulOp::Jump { target, args } => {
            let arg_strs: Vec<String> = args
                .iter()
                .map(|a| resolve_cid(*a, class_to_vreg, egraph_uf))
                .collect();
            format!("jump block{target}({})", arg_strs.join(", "))
        }
        EffectfulOp::Ret { val: Some(cid) } => {
            let val_s = resolve_cid(*cid, class_to_vreg, egraph_uf);
            format!("ret {val_s}")
        }
        EffectfulOp::Ret { val: None } => "ret".into(),
    }
}

/// Print the complete IR for a function given pre-computed printable blocks.
pub fn print_function_ir(
    func: &Function,
    blocks: &[PrintableBlock],
    class_to_vreg: &ClassVRegMap,
    egraph_uf: &crate::egraph::unionfind::UnionFind,
) -> String {
    let param_types_str: Vec<String> = func.param_types.iter().map(|t| format!("{t:?}")).collect();
    let mut out = format!("function {}({}):\n", func.name, param_types_str.join(", "));

    for block in blocks {
        // Block header with parameters
        let param_strs: Vec<String> = block
            .param_types
            .iter()
            .enumerate()
            .map(|(i, t)| format!("p{i}: {t:?}"))
            .collect();
        if param_strs.is_empty() {
            out.push_str(&format!("  block{}:\n", block.id));
        } else {
            out.push_str(&format!(
                "  block{}({}):\n",
                block.id,
                param_strs.join(", ")
            ));
        }

        // Print groups: pure ops, then barrier
        for (k, group) in block.groups.iter().enumerate() {
            // Pure ops (skip barrier pseudo-ops)
            for inst in &group.pure_ops {
                if matches!(inst.op, Op::StoreBarrier | Op::VoidCallBarrier) {
                    continue;
                }
                let op_text = fmt_op(&inst.op);
                let operand_strs: Vec<String> =
                    inst.operands.iter().map(|v| format!("v{}", v.0)).collect();
                if operand_strs.is_empty() {
                    out.push_str(&format!("    v{} = {}\n", inst.dst.0, op_text));
                } else {
                    out.push_str(&format!(
                        "    v{} = {}({})\n",
                        inst.dst.0,
                        op_text,
                        operand_strs.join(", ")
                    ));
                }
            }
            // Effectful op (barrier)
            if let Some(ref barrier) = group.barrier {
                out.push_str(&format!(
                    "    ; effectful op {k}\n    {}\n",
                    fmt_effectful(barrier, class_to_vreg, egraph_uf)
                ));
            }
        }

        // Terminator
        out.push_str(&format!(
            "    ; terminator\n    {}\n",
            fmt_effectful(&block.terminator, class_to_vreg, egraph_uf)
        ));
    }

    out
}
