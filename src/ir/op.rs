use crate::ir::condcode::CondCode;
use crate::ir::types::Type;

/// Opaque identifier for an e-class.
/// `ClassId::NONE` is used as a sentinel for absent optional operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClassId(pub u32);

impl ClassId {
    /// Sentinel value meaning "no operand" (e.g. missing index in Addr/X86Lea4).
    pub const NONE: ClassId = ClassId(u32::MAX);
}

/// Pure (side-effect-free) IR operations.
///
/// Every `Op` node has a fixed arity and a well-defined result type derivable
/// from `result_type(&self, child_types)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    // ── Arithmetic ───────────────────────────────────────────────────────────
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,

    // ── Bitwise ──────────────────────────────────────────────────────────────
    And,
    Or,
    Xor,
    /// Logical shift left
    Shl,
    /// Logical shift right
    Shr,
    /// Arithmetic shift right
    Sar,

    // ── Conversion ───────────────────────────────────────────────────────────
    /// Sign-extend child to target type.
    Sext(Type),
    /// Zero-extend child to target type.
    Zext(Type),
    /// Truncate child to target type.
    Trunc(Type),
    /// Reinterpret bits as target type.
    Bitcast(Type),

    // ── Constants ────────────────────────────────────────────────────────────
    /// Typed integer constant; the `Type` determines the e-class type.
    Iconst(i64, Type),
    /// FP constant stored as raw IEEE 754 bits (use `f64::to_bits()`).
    Fconst(u64),

    // ── Function parameters ───────────────────────────────────────────────────
    /// A function parameter. The `u32` is the zero-based parameter index.
    /// Algebraic rules and isel rules must not rewrite this op.
    Param(u32, Type),

    // ── Block parameters ──────────────────────────────────────────────────────
    /// A block parameter (SSA phi-like value). Fields: (block_id, param_idx, type).
    /// Distinct from Param to avoid collision with function parameters or Iconst sentinels.
    BlockParam(u32, u32, Type),

    // ── Comparison ───────────────────────────────────────────────────────────
    Icmp(CondCode),

    // ── Floating-point ───────────────────────────────────────────────────────
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fsqrt,

    // ── Conditional select ───────────────────────────────────────────────────
    /// `Select(flags, t, f)` — returns `t` if condition holds, else `f`.
    Select,

    // ── Projections ──────────────────────────────────────────────────────────
    /// Extract first element of a Pair.
    Proj0,
    /// Extract second element of a Pair.
    Proj1,

    // ── x86-64 machine ops ───────────────────────────────────────────────────
    /// ALU ops that set flags — produce `Pair(childtype, Flags)`.
    X86Add,
    X86Sub,
    X86And,
    X86Or,
    X86Xor,
    X86Shl,
    X86Sar,
    X86Shr,

    /// `lea [base + idx]`
    X86Lea2,
    /// `lea [base + idx * scale]` — scale embedded in op
    X86Lea3 {
        scale: u8,
    },
    /// `lea [base + idx * scale + disp]`
    X86Lea4 {
        scale: u8,
        disp: i32,
    },

    /// `imul dst, src, imm` — 3-operand signed multiply; produces `Pair(I64, Flags)`.
    X86Imul3,

    /// Conditional move — `cmov(cc, flags, t, f)` → `Pair` is not produced; returns the value type.
    X86Cmov(CondCode),

    /// Set byte from flags — `setcc` → I8.
    X86Setcc(CondCode),

    /// Addressing-mode node: `[base + index * scale + disp]`.
    /// `scale` must be 1, 2, 4, or 8. Use `ClassId::NONE` for absent index.
    Addr {
        scale: u8,
        disp: i32,
    },
}

impl Op {
    /// Derive the result type of this node given the types of its children.
    ///
    /// Panics on type mismatches or wrong child counts.
    pub fn result_type(&self, child_types: &[Type]) -> Type {
        match self {
            // ── Arithmetic (binary, same integer type) ────────────────────────
            Op::Add | Op::Sub | Op::Mul | Op::UDiv | Op::SDiv | Op::URem | Op::SRem => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, &child_types[1]
                );
                t.clone()
            }

            // ── Bitwise (binary, same integer type) ──────────────────────────
            Op::And | Op::Or | Op::Xor => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, &child_types[1]
                );
                t.clone()
            }

            // ── Shifts (two integer operands, may differ; result = first) ────
            Op::Shl | Op::Shr | Op::Sar => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert!(
                    child_types[0].is_integer(),
                    "{self:?} first operand must be integer, got {:?}",
                    child_types[0]
                );
                assert!(
                    child_types[1].is_integer(),
                    "{self:?} shift amount must be integer, got {:?}",
                    child_types[1]
                );
                child_types[0].clone()
            }

            // ── Conversion (1 child, target type embedded) ───────────────────
            Op::Sext(target) => {
                assert_eq!(child_types.len(), 1, "Sext requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "Sext requires integer child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Zext(target) => {
                assert_eq!(child_types.len(), 1, "Zext requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "Zext requires integer child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Trunc(target) => {
                assert_eq!(child_types.len(), 1, "Trunc requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "Trunc requires integer child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Bitcast(target) => {
                assert_eq!(child_types.len(), 1, "Bitcast requires 1 child");
                target.clone()
            }

            // ── Constants (0 children) ────────────────────────────────────────
            Op::Iconst(_val, ty) => {
                assert_eq!(child_types.len(), 0, "Iconst requires 0 children");
                ty.clone()
            }
            Op::Fconst(_) => {
                assert_eq!(child_types.len(), 0, "Fconst requires 0 children");
                Type::F64
            }
            Op::Param(_idx, ty) => {
                assert_eq!(child_types.len(), 0, "Param requires 0 children");
                ty.clone()
            }
            Op::BlockParam(_block_id, _param_idx, ty) => {
                assert_eq!(child_types.len(), 0, "BlockParam requires 0 children");
                ty.clone()
            }

            // ── Comparison ────────────────────────────────────────────────────
            Op::Icmp(_cc) => {
                assert_eq!(child_types.len(), 2, "Icmp requires 2 children");
                let t = &child_types[0];
                assert!(t.is_integer(), "Icmp requires integer operands, got {t:?}");
                assert_eq!(
                    &child_types[1], t,
                    "Icmp operand type mismatch: {:?} vs {:?}",
                    t, &child_types[1]
                );
                Type::Flags
            }

            // ── FP binary ops ─────────────────────────────────────────────────
            Op::Fadd | Op::Fsub | Op::Fmul | Op::Fdiv => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert_eq!(child_types[0], Type::F64, "{self:?} requires F64 operands");
                assert_eq!(child_types[1], Type::F64, "{self:?} requires F64 operands");
                Type::F64
            }
            Op::Fsqrt => {
                assert_eq!(child_types.len(), 1, "Fsqrt requires 1 child");
                assert_eq!(child_types[0], Type::F64, "Fsqrt requires F64 operand");
                Type::F64
            }

            // ── Select ────────────────────────────────────────────────────────
            Op::Select => {
                assert_eq!(
                    child_types.len(),
                    3,
                    "Select requires 3 children (flags, t, f)"
                );
                assert_eq!(
                    child_types[0],
                    Type::Flags,
                    "Select first child must be Flags"
                );
                assert_eq!(
                    child_types[1], child_types[2],
                    "Select true/false branches must have same type: {:?} vs {:?}",
                    child_types[1], child_types[2]
                );
                child_types[1].clone()
            }

            // ── Projections ───────────────────────────────────────────────────
            Op::Proj0 => {
                assert_eq!(child_types.len(), 1, "Proj0 requires 1 child");
                match &child_types[0] {
                    Type::Pair(a, _b) => *a.clone(),
                    other => panic!("Proj0 requires Pair child, got {other:?}"),
                }
            }
            Op::Proj1 => {
                assert_eq!(child_types.len(), 1, "Proj1 requires 1 child");
                match &child_types[0] {
                    Type::Pair(_a, b) => *b.clone(),
                    other => panic!("Proj1 requires Pair child, got {other:?}"),
                }
            }

            // ── x86 ALU (binary integer → Pair(childtype, Flags)) ────────────
            Op::X86Add | Op::X86Sub | Op::X86And | Op::X86Or | Op::X86Xor => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, &child_types[1]
                );
                Type::Pair(Box::new(t.clone()), Box::new(Type::Flags))
            }
            Op::X86Shl | Op::X86Sar | Op::X86Shr => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert!(
                    child_types[0].is_integer(),
                    "{self:?} first operand must be integer, got {:?}",
                    child_types[0]
                );
                assert!(
                    child_types[1].is_integer(),
                    "{self:?} shift amount must be integer, got {:?}",
                    child_types[1]
                );
                Type::Pair(Box::new(child_types[0].clone()), Box::new(Type::Flags))
            }

            // ── x86 LEA variants (I64, I64 → I64) ───────────────────────────
            Op::X86Lea2 | Op::X86Lea3 { .. } | Op::X86Lea4 { .. } => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert_eq!(child_types[0], Type::I64, "{self:?} requires I64 base");
                assert_eq!(child_types[1], Type::I64, "{self:?} requires I64 index");
                Type::I64
            }

            // ── X86Imul3 (2 children → Pair(I64, Flags)) ────────────────────
            Op::X86Imul3 => {
                assert_eq!(child_types.len(), 2, "X86Imul3 requires 2 children");
                assert!(
                    child_types[0].is_integer(),
                    "X86Imul3 first operand must be integer, got {:?}",
                    child_types[0]
                );
                assert!(
                    child_types[1].is_integer(),
                    "X86Imul3 second operand must be integer, got {:?}",
                    child_types[1]
                );
                Type::Pair(Box::new(Type::I64), Box::new(Type::Flags))
            }

            // ── X86Cmov (flags, t, f → t's type) ────────────────────────────
            Op::X86Cmov(_cc) => {
                assert_eq!(
                    child_types.len(),
                    3,
                    "X86Cmov requires 3 children (flags, t, f)"
                );
                assert_eq!(
                    child_types[0],
                    Type::Flags,
                    "X86Cmov first child must be Flags"
                );
                assert_eq!(
                    child_types[1], child_types[2],
                    "X86Cmov true/false branches must have same type: {:?} vs {:?}",
                    child_types[1], child_types[2]
                );
                child_types[1].clone()
            }

            // ── X86Setcc (flags → I8) ─────────────────────────────────────────
            Op::X86Setcc(_cc) => {
                assert_eq!(child_types.len(), 1, "X86Setcc requires 1 child");
                assert_eq!(child_types[0], Type::Flags, "X86Setcc child must be Flags");
                Type::I8
            }

            // ── Addr (base I64, index I64 → I64) ─────────────────────────────
            Op::Addr { .. } => {
                assert_eq!(
                    child_types.len(),
                    2,
                    "Addr requires 2 children (base, index)"
                );
                assert_eq!(child_types[0], Type::I64, "Addr base must be I64");
                assert_eq!(child_types[1], Type::I64, "Addr index must be I64");
                Type::I64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::condcode::CondCode;
    use crate::ir::types::Type;

    // ── Pure arithmetic ───────────────────────────────────────────────────────

    #[test]
    fn add_i32() {
        let ty = Op::Add.result_type(&[Type::I32, Type::I32]);
        assert_eq!(ty, Type::I32);
    }

    #[test]
    fn add_i64() {
        let ty = Op::Add.result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    #[should_panic]
    fn add_type_mismatch() {
        Op::Add.result_type(&[Type::I32, Type::I64]);
    }

    #[test]
    #[should_panic]
    fn add_float_rejected() {
        Op::Add.result_type(&[Type::F64, Type::F64]);
    }

    #[test]
    fn sub_i64() {
        assert_eq!(Op::Sub.result_type(&[Type::I64, Type::I64]), Type::I64);
    }

    #[test]
    fn mul_i16() {
        assert_eq!(Op::Mul.result_type(&[Type::I16, Type::I16]), Type::I16);
    }

    #[test]
    fn udiv_i32() {
        assert_eq!(Op::UDiv.result_type(&[Type::I32, Type::I32]), Type::I32);
    }

    #[test]
    fn urem_i8() {
        assert_eq!(Op::URem.result_type(&[Type::I8, Type::I8]), Type::I8);
    }

    // ── Bitwise ───────────────────────────────────────────────────────────────

    #[test]
    fn and_i64() {
        assert_eq!(Op::And.result_type(&[Type::I64, Type::I64]), Type::I64);
    }

    #[test]
    fn xor_i32() {
        assert_eq!(Op::Xor.result_type(&[Type::I32, Type::I32]), Type::I32);
    }

    #[test]
    fn shl_different_widths() {
        // shift amount can differ from value type
        let ty = Op::Shl.result_type(&[Type::I64, Type::I8]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    fn sar_i32() {
        assert_eq!(Op::Sar.result_type(&[Type::I32, Type::I32]), Type::I32);
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    #[test]
    fn sext_i32_to_i64() {
        assert_eq!(Op::Sext(Type::I64).result_type(&[Type::I32]), Type::I64);
    }

    #[test]
    fn zext_i8_to_i64() {
        assert_eq!(Op::Zext(Type::I64).result_type(&[Type::I8]), Type::I64);
    }

    #[test]
    fn trunc_i64_to_i32() {
        assert_eq!(Op::Trunc(Type::I32).result_type(&[Type::I64]), Type::I32);
    }

    #[test]
    fn bitcast_i64_to_f64() {
        assert_eq!(Op::Bitcast(Type::F64).result_type(&[Type::I64]), Type::F64);
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    #[test]
    fn iconst_i64() {
        assert_eq!(Op::Iconst(42, Type::I64).result_type(&[]), Type::I64);
    }

    #[test]
    fn iconst_i32() {
        assert_eq!(Op::Iconst(0, Type::I32).result_type(&[]), Type::I32);
    }

    #[test]
    fn fconst_is_f64() {
        assert_eq!(Op::Fconst(0u64).result_type(&[]), Type::F64);
    }

    // ── Comparison ────────────────────────────────────────────────────────────

    #[test]
    fn icmp_produces_flags() {
        assert_eq!(
            Op::Icmp(CondCode::Slt).result_type(&[Type::I64, Type::I64]),
            Type::Flags
        );
    }

    #[test]
    fn icmp_eq_i32() {
        assert_eq!(
            Op::Icmp(CondCode::Eq).result_type(&[Type::I32, Type::I32]),
            Type::Flags
        );
    }

    #[test]
    #[should_panic]
    fn icmp_type_mismatch() {
        Op::Icmp(CondCode::Eq).result_type(&[Type::I32, Type::I64]);
    }

    // ── FP ops ────────────────────────────────────────────────────────────────

    #[test]
    fn fadd_f64() {
        assert_eq!(Op::Fadd.result_type(&[Type::F64, Type::F64]), Type::F64);
    }

    #[test]
    fn fsqrt_f64() {
        assert_eq!(Op::Fsqrt.result_type(&[Type::F64]), Type::F64);
    }

    #[test]
    #[should_panic]
    fn fadd_wrong_type() {
        Op::Fadd.result_type(&[Type::I64, Type::I64]);
    }

    // ── Select ────────────────────────────────────────────────────────────────

    #[test]
    fn select_i64() {
        let ty = Op::Select.result_type(&[Type::Flags, Type::I64, Type::I64]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    #[should_panic]
    fn select_branch_mismatch() {
        Op::Select.result_type(&[Type::Flags, Type::I32, Type::I64]);
    }

    // ── Projections ───────────────────────────────────────────────────────────

    #[test]
    fn proj0_pair() {
        let pair = Type::Pair(Box::new(Type::I64), Box::new(Type::Flags));
        assert_eq!(Op::Proj0.result_type(&[pair]), Type::I64);
    }

    #[test]
    fn proj1_pair() {
        let pair = Type::Pair(Box::new(Type::I64), Box::new(Type::Flags));
        assert_eq!(Op::Proj1.result_type(&[pair]), Type::Flags);
    }

    #[test]
    #[should_panic]
    fn proj0_non_pair() {
        Op::Proj0.result_type(&[Type::I64]);
    }

    // ── x86-64 machine ops ────────────────────────────────────────────────────

    #[test]
    fn x86add_produces_pair() {
        let ty = Op::X86Add.result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86sub_produces_pair() {
        let ty = Op::X86Sub.result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86and_i32() {
        let ty = Op::X86And.result_type(&[Type::I32, Type::I32]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I32), Box::new(Type::Flags)));
    }

    #[test]
    fn x86shl_produces_pair() {
        let ty = Op::X86Shl.result_type(&[Type::I64, Type::I8]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86lea2_i64() {
        assert_eq!(Op::X86Lea2.result_type(&[Type::I64, Type::I64]), Type::I64);
    }

    #[test]
    fn x86lea3_i64() {
        assert_eq!(
            Op::X86Lea3 { scale: 2 }.result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    fn x86lea4_i64() {
        assert_eq!(
            Op::X86Lea4 { scale: 4, disp: 16 }.result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    fn x86imul3_pair() {
        let ty = Op::X86Imul3.result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86cmov_i64() {
        let ty = Op::X86Cmov(CondCode::Ne).result_type(&[Type::Flags, Type::I64, Type::I64]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    fn x86setcc_i8() {
        assert_eq!(
            Op::X86Setcc(CondCode::Eq).result_type(&[Type::Flags]),
            Type::I8
        );
    }

    #[test]
    fn addr_i64() {
        assert_eq!(
            Op::Addr { scale: 4, disp: 8 }.result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    #[should_panic]
    fn x86add_flags_rejected() {
        Op::X86Add.result_type(&[Type::Flags, Type::Flags]);
    }

    #[test]
    #[should_panic]
    fn x86cmov_wrong_flags() {
        Op::X86Cmov(CondCode::Eq).result_type(&[Type::I64, Type::I64, Type::I64]);
    }

    // ── ClassId sentinel ──────────────────────────────────────────────────────

    #[test]
    fn classid_none_is_max() {
        assert_eq!(ClassId::NONE, ClassId(u32::MAX));
    }
}
