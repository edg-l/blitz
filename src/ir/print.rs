use crate::egraph::extract::ClassVRegMap;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, MachOp, Op, PseudoOp, PureOp};
use crate::ir::types::Type;
use crate::schedule::scheduler::ScheduledInst;

/// Format a single Op variant as a human-readable string.
pub fn fmt_op(op: &Op) -> String {
    match op {
        // Arithmetic
        Op::Pure(PureOp::Add) => "add".into(),
        Op::Pure(PureOp::Sub) => "sub".into(),
        Op::Pure(PureOp::Mul) => "mul".into(),
        Op::Pure(PureOp::UDiv) => "udiv".into(),
        Op::Pure(PureOp::SDiv) => "sdiv".into(),
        Op::Pure(PureOp::URem) => "urem".into(),
        Op::Pure(PureOp::SRem) => "srem".into(),

        // Bitwise
        Op::Pure(PureOp::And) => "and".into(),
        Op::Pure(PureOp::Or) => "or".into(),
        Op::Pure(PureOp::Xor) => "xor".into(),
        Op::Pure(PureOp::Shl) => "shl".into(),
        Op::Pure(PureOp::Shr) => "shr".into(),
        Op::Pure(PureOp::Sar) => "sar".into(),

        // Conversion
        Op::Pure(PureOp::Sext(ty)) => format!("sext({ty:?})"),
        Op::Pure(PureOp::Zext(ty)) => format!("zext({ty:?})"),
        Op::Pure(PureOp::Trunc(ty)) => format!("trunc({ty:?})"),
        Op::Pure(PureOp::Bitcast(ty)) => format!("bitcast({ty:?})"),

        // Constants
        Op::Pure(PureOp::Iconst(val, ty)) => format!("iconst({val}, {ty:?})"),
        Op::Pure(PureOp::Fconst(bits, ty)) => format!("fconst(0x{bits:x}, {ty:?})"),

        // Parameters
        Op::Pure(PureOp::Param(idx, ty)) => format!("param({idx}, {ty:?})"),
        Op::Pure(PureOp::BlockParam(bid, pidx, ty)) => {
            format!("block_param(b{bid}, {pidx}, {ty:?})")
        }

        // Comparison
        Op::Pure(PureOp::Icmp(cc)) => format!("icmp({cc:?})"),
        Op::Pure(PureOp::Fcmp(cc)) => format!("fcmp({cc:?})"),

        // Float/int conversions
        Op::Pure(PureOp::IntToFloat(ty)) => format!("int_to_float({ty:?})"),
        Op::Pure(PureOp::FloatToInt(ty)) => format!("float_to_int({ty:?})"),
        Op::Pure(PureOp::FloatExt) => "float_ext".into(),
        Op::Pure(PureOp::FloatTrunc) => "float_trunc".into(),

        // FP ops
        Op::Pure(PureOp::Fadd) => "fadd".into(),
        Op::Pure(PureOp::Fsub) => "fsub".into(),
        Op::Pure(PureOp::Fmul) => "fmul".into(),
        Op::Pure(PureOp::Fdiv) => "fdiv".into(),
        Op::Pure(PureOp::Fsqrt) => "fsqrt".into(),

        // Select
        Op::Pure(PureOp::Select) => "select".into(),

        // Projections
        Op::Pure(PureOp::Proj0) => "proj0".into(),
        Op::Pure(PureOp::Proj1) => "proj1".into(),

        // x86 ALU
        Op::Mach(MachOp::X86Add) => "x86_add".into(),
        Op::Mach(MachOp::X86Sub) => "x86_sub".into(),
        Op::Mach(MachOp::X86And) => "x86_and".into(),
        Op::Mach(MachOp::X86Or) => "x86_or".into(),
        Op::Mach(MachOp::X86Xor) => "x86_xor".into(),
        Op::Mach(MachOp::X86Shl) => "x86_shl".into(),
        Op::Mach(MachOp::X86Sar) => "x86_sar".into(),
        Op::Mach(MachOp::X86Shr) => "x86_shr".into(),

        // x86 immediate shifts
        Op::Mach(MachOp::X86ShlImm(n)) => format!("x86_shl_imm({n})"),
        Op::Mach(MachOp::X86ShrImm(n)) => format!("x86_shr_imm({n})"),
        Op::Mach(MachOp::X86SarImm(n)) => format!("x86_sar_imm({n})"),

        // x86 flag-only compare with immediate
        Op::Mach(MachOp::X86CmpI { imm, ty }) => format!("x86_cmp_imm({imm}, {ty:?})"),

        // x86 LEA
        Op::Mach(MachOp::X86Lea2) => "x86_lea2".into(),
        Op::Mach(MachOp::X86Lea3 { scale }) => format!("x86_lea3(scale={scale})"),
        Op::Mach(MachOp::X86Lea4 { scale, disp }) => {
            format!("x86_lea4(scale={scale}, disp={disp})")
        }

        // x86 multiply/divide
        Op::Mach(MachOp::X86Imul3) => "x86_imul3".into(),
        Op::Mach(MachOp::X86Idiv(..)) => "x86_idiv".into(),
        Op::Mach(MachOp::X86Div(..)) => "x86_div".into(),

        // x86 conditional ops
        Op::Mach(MachOp::X86Cmov(cc)) => format!("x86_cmov({cc:?})"),
        Op::Mach(MachOp::X86Setcc(cc)) => format!("x86_setcc({cc:?})"),

        // Addressing
        Op::Pure(PureOp::Addr { scale, disp }) => format!("addr(scale={scale}, disp={disp})"),

        // Load/Call result placeholders
        Op::Pseudo(PseudoOp::LoadResult(id, ty)) => format!("load_result({id}, {ty:?})"),
        Op::Pseudo(PseudoOp::CallResult(id, ty)) => format!("call_result({id}, {ty:?})"),

        // x86 FP double
        Op::Mach(MachOp::X86Addsd) => "x86_addsd".into(),
        Op::Mach(MachOp::X86Subsd) => "x86_subsd".into(),
        Op::Mach(MachOp::X86Mulsd) => "x86_mulsd".into(),
        Op::Mach(MachOp::X86Divsd) => "x86_divsd".into(),
        Op::Mach(MachOp::X86Sqrtsd) => "x86_sqrtsd".into(),

        // x86 FP single
        Op::Mach(MachOp::X86Addss) => "x86_addss".into(),
        Op::Mach(MachOp::X86Subss) => "x86_subss".into(),
        Op::Mach(MachOp::X86Mulss) => "x86_mulss".into(),
        Op::Mach(MachOp::X86Divss) => "x86_divss".into(),
        Op::Mach(MachOp::X86Sqrtss) => "x86_sqrtss".into(),

        // x86 FP conversion
        Op::Mach(MachOp::X86Cvtsi2sd(ty)) => format!("x86_cvtsi2sd({ty:?})"),
        Op::Mach(MachOp::X86Cvtsi2ss(ty)) => format!("x86_cvtsi2ss({ty:?})"),
        Op::Mach(MachOp::X86Cvttsd2si(ty)) => format!("x86_cvttsd2si({ty:?})"),
        Op::Mach(MachOp::X86Cvttss2si(ty)) => format!("x86_cvttss2si({ty:?})"),
        Op::Mach(MachOp::X86Cvtsd2ss) => "x86_cvtsd2ss".into(),
        Op::Mach(MachOp::X86Cvtss2sd) => "x86_cvtss2sd".into(),

        // x86 FP comparison
        Op::Mach(MachOp::X86Ucomisd) => "x86_ucomisd".into(),
        Op::Mach(MachOp::X86UcomisdCc(cc)) => format!("x86_ucomisd_cc({cc:?})"),
        Op::Mach(MachOp::X86UcomissCc(cc)) => format!("x86_ucomiss_cc({cc:?})"),
        Op::Mach(MachOp::X86Ucomiss) => "x86_ucomiss".into(),

        // Stack address
        Op::Pseudo(PseudoOp::StackAddr(slot)) => format!("stack_addr({slot})"),

        // Global address
        Op::Pseudo(PseudoOp::GlobalAddr(name)) => format!("global_addr(\"{}\")", name),

        // x86 conversion ops
        Op::Mach(MachOp::X86Movsx { from, to }) => format!("x86_movsx({from:?} -> {to:?})"),
        Op::Mach(MachOp::X86Movzx { from, to }) => format!("x86_movzx({from:?} -> {to:?})"),
        Op::Mach(MachOp::X86Trunc { from, to }) => format!("x86_trunc({from:?} -> {to:?})"),
        Op::Mach(MachOp::X86Bitcast { from, to }) => format!("x86_bitcast({from:?} -> {to:?})"),

        // Spill ops
        Op::Pseudo(PseudoOp::SpillStore(s)) => format!("spill_store({s})"),
        Op::Pseudo(PseudoOp::SpillLoad(s)) => format!("spill_load({s})"),
        Op::Pseudo(PseudoOp::XmmSpillStore(s)) => format!("xmm_spill_store({s})"),
        Op::Pseudo(PseudoOp::XmmSpillLoad(s)) => format!("xmm_spill_load({s})"),

        // Barrier pseudo-ops
        Op::Pseudo(PseudoOp::StoreBarrier) => "store_barrier".into(),
        Op::Pseudo(PseudoOp::TerminatorArgs(_)) => "terminator_args".into(),
        Op::Pseudo(PseudoOp::VoidCallBarrier) => "void_call_barrier".into(),
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
            let addr_s = resolve_cid(addr.class(), class_to_vreg, egraph_uf);
            let result_s = resolve_cid(result.class(), class_to_vreg, egraph_uf);
            format!("load {ty:?} {addr_s} -> {result_s}")
        }
        EffectfulOp::Store { addr, val, ty } => {
            let addr_s = resolve_cid(addr.class(), class_to_vreg, egraph_uf);
            let val_s = resolve_cid(val.class(), class_to_vreg, egraph_uf);
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
                .map(|a| resolve_cid(a.class(), class_to_vreg, egraph_uf))
                .collect();
            let result_strs: Vec<String> = results
                .iter()
                .map(|r| resolve_cid(r.class(), class_to_vreg, egraph_uf))
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
            let cond_s = resolve_cid(cond.class(), class_to_vreg, egraph_uf);
            let true_arg_strs: Vec<String> = true_args
                .class_ids()
                .map(|a| resolve_cid(a, class_to_vreg, egraph_uf))
                .collect();
            let false_arg_strs: Vec<String> = false_args
                .class_ids()
                .map(|a| resolve_cid(a, class_to_vreg, egraph_uf))
                .collect();
            format!(
                "branch {cc:?} {cond_s} block{bb_true}({}) block{bb_false}({})",
                true_arg_strs.join(", "),
                false_arg_strs.join(", ")
            )
        }
        EffectfulOp::Jump { target, args } => {
            let arg_strs: Vec<String> = args
                .class_ids()
                .map(|a| resolve_cid(a, class_to_vreg, egraph_uf))
                .collect();
            format!("jump block{target}({})", arg_strs.join(", "))
        }
        EffectfulOp::Ret { val: Some(cid) } => {
            let val_s = resolve_cid(cid.class(), class_to_vreg, egraph_uf);
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
                if matches!(
                    inst.op,
                    Op::Pseudo(PseudoOp::StoreBarrier)
                        | Op::Pseudo(PseudoOp::VoidCallBarrier)
                        | Op::Pseudo(PseudoOp::TerminatorArgs(_))
                ) {
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
