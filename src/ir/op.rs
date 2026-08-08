use crate::ir::condcode::CondCode;
use crate::ir::types::Type;
use crate::x86::reg::RegClass;

/// Opaque identifier for an e-class.
/// `ClassId::NONE` is used as a sentinel for absent optional operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassId(pub u32);

impl ClassId {
    /// Sentinel value meaning "no operand" (e.g. missing index in Addr/X86Lea4).
    pub const NONE: ClassId = ClassId(u32::MAX);
}

/// Generic, target-independent IR operations.
///
/// Every one of these **must be lowered by the e-graph's isel phases before
/// extraction**: `cost.rs` prices the whole type at infinity and `lower.rs`
/// rejects it, each with one arm rather than a list of variants that can drift
/// apart. That drift is why this type exists -- the two lists were enumerated
/// separately and had already disagreed about `Fcmp`.
///
/// Arity is fixed per variant and the result type is derivable from
/// `Op::result_type`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PureOp {
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
    /// FP constant stored as raw IEEE 754 bits (use `f64::to_bits()` / `f32::to_bits()`).
    /// The `Type` determines F32 vs F64.
    Fconst(u64, Type),

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
    /// Float comparison: takes 2 float children, produces `Type::Flags`.
    Fcmp(CondCode),

    // ── Float/int conversion ───────────────────────────────────────────────
    /// Signed integer -> float. Type param is target float type (F32 or F64).
    IntToFloat(Type),
    /// Float -> signed integer (truncation). Type param is target int type (I32 or I64).
    FloatToInt(Type),
    /// F32 -> F64 extension.
    FloatExt,
    /// F64 -> F32 truncation.
    FloatTrunc,

    // ── Floating-point ───────────────────────────────────────────────────────
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fsqrt,

    // ── Conditional select ───────────────────────────────────────────────────
    /// `Select(flags, t, f)` — returns `t` if condition holds, else `f`.
    /// Select the `t` operand when `cc` holds of the flags, else `f`.
    ///
    /// The condition code rides on the node for the same reason it rides on
    /// `EffectfulOp::Branch`: flags are shared between comparisons, so the
    /// flags operand cannot say which condition this select tests. `a == b`
    /// and `a != b` set identical flags and their `Icmp` classes merge onto one
    /// shared compare; recovering the cc from that class afterwards returns
    /// whichever node came first.
    Select(CondCode),

    // ── Projections ──────────────────────────────────────────────────────────
    /// Extract first element of a Pair.
    Proj0,
    /// Extract second element of a Pair.
    Proj1,

    /// Addressing-mode node: `[base + index * scale + disp]`.
    /// `scale` must be 1, 2, 4, or 8. Use `ClassId::NONE` for absent index.
    Addr {
        scale: u8,
        disp: i32,
    },
}

/// x86-64 machine operations: what instruction selection produces and what
/// `lower.rs` turns into `MachInst`. One per instruction form the backend emits.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MachOp {
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

    /// Flag-only compare with immediate — `cmp reg, imm` (or `test reg, reg` when
    /// imm is 0); 1 child; produces `Flags` directly (no `Pair`). Emitted by isel
    /// when an `Icmp` has an `Iconst` RHS that fits in i32, avoiding the
    /// destructive `mov + sub` sequence and the separate iconst materialization.
    ///
    /// `ty` is the operand's integer type — needed at lowering time because
    /// the op produces `Flags` (no size info in the dst class) and the operand
    /// may be a spill-reload vreg whose type isn't tracked in `vreg_types`.
    X86CmpI {
        imm: i32,
        ty: Type,
    },

    /// ALU op with a baked-in immediate — `add dst, imm` and friends; 1 child;
    /// produces `Pair(childtype, Flags)` exactly as the register form does, so
    /// `Proj1` of one is the same flags a register form would set.
    ///
    /// Emitted by isel when a `Proj0` of an ALU pair has an `Iconst` operand
    /// that fits in `i32`, which saves both the register the constant would
    /// occupy and the `mov` that materializes it. On 64-bit operands the
    /// immediate is sign-extended from 32 bits, which is what `i32::try_from`
    /// on the constant already guarantees.
    ///
    /// No width field, unlike `X86CmpI`: the result is a pair whose first
    /// element is the operand type, so the dst class carries the width.
    X86AddI(i32),
    X86SubI(i32),
    /// `imul dst, src, imm` -- the 3-operand signed multiply; 1 child;
    /// produces `Pair(childtype, Flags)` as the register form does.
    ///
    /// **Not two-address**, which is the whole of why it is worth a node: the
    /// register form is `mov dst, a` then `imul dst, b`, and the constant needs
    /// a `mov` of its own before either. This reads its operand and writes a
    /// different register in one instruction.
    X86ImulI(i32),
    X86AndI(i32),
    X86OrI(i32),
    X86XorI(i32),

    /// Shift left by immediate count — `shl dst, imm`; 1 child. Free of RCX pressure.
    X86ShlImm(u8),
    /// Logical shift right by immediate count — `shr dst, imm`; 1 child.
    X86ShrImm(u8),
    /// Arithmetic shift right by immediate count — `sar dst, imm`; 1 child.
    X86SarImm(u8),

    /// Rotate left by immediate count — `rol dst, imm`; 1 child. Emitted by isel
    /// for `Or(Shl(x, k), Shr(x, w - k))` on a `w`-bit `x`, which is one
    /// instruction instead of three and leaves `x` with a single use.
    ///
    /// There is no right-rotate form: `ror k` is `rol (w - k)` in the same
    /// encoding size, and a right rotate written in the source reaches isel as
    /// the same `Or` of two shifts, whose operand order says nothing about which
    /// direction was meant.
    X86RolImm(u8),

    /// Double shift left by immediate count -- `shld dst, src, imm`; 2 children.
    /// `dst` is shifted left by `imm` and the vacated low bits are filled from
    /// the high end of `src`, which is left unchanged. Emitted by isel for
    /// `Or(Shl(x, k), Shr(y, w - k))` on `w`-bit `x` and `y`, one instruction
    /// where the shift pair is three.
    ///
    /// There is no `shrd` form: `shrd y, x, w - k` computes the same bits as
    /// `shld x, y, k`, so the two differ only in which operand the destructive
    /// form consumes, and extraction has no basis to prefer one -- both price
    /// the same and neither knows which operand dies here.
    X86ShldImm(u8),

    /// Set bit `n` of `x` -- `bts x, n`; 2 children `(x, n)`. Emitted by isel
    /// for `Or(x, Shl(1, n))`, where a variable `n` otherwise costs a constant
    /// materialization, a shift through CL and the `or` itself.
    ///
    /// x86 takes the bit index modulo the operand width, and a `1 << n` that
    /// reached here with `n` at or past the width is undefined in C, so the two
    /// agree everywhere the source is defined. There is no byte form.
    X86Bts,
    /// Clear bit `n` of `x` -- `btr x, n`; 2 children. Isel for
    /// `And(x, Xor(Shl(1, n), -1))`.
    X86Btr,
    /// Complement bit `n` of `x` -- `btc x, n`; 2 children. Isel for
    /// `Xor(x, Shl(1, n))`.
    X86Btc,

    /// The same three with a constant bit index -- `bts x, imm8`; 1 child.
    /// Emitted by isel for `Or`/`Xor`/`And` against a one-bit constant mask (or
    /// its complement) whose bit sits at 7 or above, which is where the
    /// immediate-form ALU stops reaching and the register form has to
    /// materialize the mask.
    X86BtsI(u8),
    X86BtrI(u8),
    X86BtcI(u8),

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

    /// `imul dst, src, imm` — 3-operand signed multiply; produces `Pair(<int_ty>, Flags)`.
    X86Imul3,

    /// Signed integer division: `idiv` — takes (dividend, divisor), produces
    /// `Pair(<int_ty>, <int_ty>)` where Proj0 = quotient (RAX), Proj1 = remainder (RDX).
    ///
    /// Hardware notes:
    /// - Division by zero raises SIGFPE (x86 #DE exception).
    /// - INT_MIN / -1 raises SIGFPE (overflow). Matches C undefined behavior.
    ///
    /// The operand type rides on the op because the width cannot be recovered
    /// later: `vreg_types` is built before the splitter runs, so a reload or a
    /// re-emitted copy has no entry there and lowering fell back to 64 bits. A
    /// 64-bit `idiv` reading a negative 32-bit divisor materialized by `mov
    /// ecx,imm32` divides by its zero-extension instead.
    X86Idiv(Type),

    /// Unsigned integer division: `div` — takes (dividend, divisor), produces
    /// `Pair(<int_ty>, <int_ty>)` where Proj0 = quotient (RAX), Proj1 = remainder (RDX).
    ///
    /// Hardware notes:
    /// - Division by zero raises SIGFPE (x86 #DE exception).
    ///
    /// Carries its operand type for the same reason as `X86Idiv`.
    X86Div(Type),

    /// Conditional move — `cmov(cc, flags, t, f)` → `Pair` is not produced; returns the value type.
    X86Cmov(CondCode),

    /// Set byte from flags — `setcc` → I8.
    X86Setcc(CondCode),

    /// `sbb r, r` — the carry flag broadcast over the whole register: `-CF`,
    /// so all ones below and zero at or above. Carries its result type, since
    /// its only child is `Flags` and so says nothing about the width.
    X86SbbSelf(Type),

    // ── x86-64 FP machine ops ─────────────────────────────────────────────────
    /// `addsd dst, src` — f64 + f64 → f64.
    X86Addsd,
    /// `subsd dst, src` — f64 - f64 → f64.
    X86Subsd,
    /// `mulsd dst, src` — f64 * f64 → f64.
    X86Mulsd,
    /// `divsd dst, src` — f64 / f64 → f64.
    X86Divsd,
    /// `sqrtsd dst, src` — sqrt(f64) → f64.
    X86Sqrtsd,

    /// `addss dst, src` — f32 + f32 → f32.
    X86Addss,
    /// `subss dst, src` — f32 - f32 → f32.
    X86Subss,
    /// `mulss dst, src` — f32 * f32 → f32.
    X86Mulss,
    /// `divss dst, src` — f32 / f32 → f32.
    X86Divss,
    /// `sqrtss dst, src` — sqrt(f32) → f32.
    X86Sqrtss,

    // ── x86-64 FP conversion ops ────────────────────────────────────────────────
    /// `cvtsi2sd` — signed int -> f64; 1 child (GPR), result F64.
    ///
    /// The `Type` is the *source* width and selects the 32- or 64-bit form.
    /// It must be `I32` or `I64`: the instruction has no narrower encoding, so
    /// isel sign-extends anything smaller first. Reading a 32-bit value with
    /// the 64-bit form converts the caller's leftover high bits, which SysV
    /// leaves undefined.
    X86Cvtsi2sd(Type),
    /// `cvtsi2ss` — signed int -> f32; 1 child (GPR), result F32.
    /// The `Type` is the source width; see [`MachOp::X86Cvtsi2sd`].
    X86Cvtsi2ss(Type),
    /// `cvttsd2si` — f64 -> signed int (truncation); 1 child (XMM), result = Type param.
    X86Cvttsd2si(Type),
    /// `cvttss2si` — f32 -> signed int (truncation); 1 child (XMM), result = Type param.
    X86Cvttss2si(Type),
    /// `cvtsd2ss` — f64 -> f32; 1 child (XMM), result F32.
    X86Cvtsd2ss,
    /// `cvtss2sd` — f32 -> f64; 1 child (XMM), result F64.
    X86Cvtss2sd,

    // ── x86-64 FP comparison ops (Op variants for isel) ──────────────────────
    /// `ucomisd` — compare two f64 values, sets flags; 2 children, result Flags.
    X86Ucomisd,
    /// `ucomiss` — compare two f32 values, sets flags; 2 children, result Flags.
    X86Ucomiss,
    /// `ucomisd` for a **composite** condition code; 2 children, result Flags.
    ///
    /// `OrdEq` is ordered *and* equal (`ZF=1 ∧ PF=0`) and `UnordNe` is unordered
    /// *or* not-equal (`ZF=0 ∨ PF=1`): no single `setcc`/`jcc` encodes either, so
    /// they cannot ride the shared `X86Ucomisd` node the way one-test codes do.
    /// The code rides on the node so hashconsing keeps `(OrdEq, a, b)` distinct
    /// from any other comparison of the same pair.
    X86UcomisdCc(CondCode),
    /// `ucomiss` for a composite condition code. See [`MachOp::X86UcomisdCc`].
    X86UcomissCc(CondCode),

    // ── x86-64 conversion ops ─────────────────────────────────────────────────
    /// `movsx` — sign-extend from `from` type to `to` type; 1 child.
    X86Movsx {
        from: Type,
        to: Type,
    },
    /// `movzx` — zero-extend from `from` type to `to` type; 1 child.
    X86Movzx {
        from: Type,
        to: Type,
    },
    /// Truncate from `from` type to `to` type; 1 child. Free on x86-64.
    X86Trunc {
        from: Type,
        to: Type,
    },
    /// Reinterpret bits without conversion; 1 child.
    /// - int->float or float->int: MOVQ between GPR and XMM.
    /// - same class, same size: register copy (or no-op if coalesced).
    X86Bitcast {
        from: Type,
        to: Type,
    },
}

/// Schedule-level pseudo operations.
///
/// These are neither IR nor machine instructions: they carry spill traffic,
/// barrier liveness and terminator arguments through scheduling and allocation.
/// Several define no value at all (`Op::has_no_result`), and a structure that
/// assumes an op defines one is where every bug in this area has come from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PseudoOp {
    // ── Stack slot address ─────────────────────────────────────────────────────
    /// Address of stack slot N. Zero children, returns I64.
    /// Lowered to an LEA from the frame pointer.
    StackAddr(u32),

    // ── Global variable address ───────────────────────────────────────────────
    /// Address of a global variable by name. Zero children, returns I64.
    /// Lowered to LEA [RIP + symbol].
    GlobalAddr(String),

    // ── Load result placeholder ───────────────────────────────────────────────
    /// Placeholder node representing the result of a Load effectful op.
    /// The `u32` is a globally unique ID (from `FunctionBuilder::next_uid`) to
    /// ensure each load gets a distinct e-class. Has no children.
    LoadResult(u32, Type),

    // ── Call result placeholder ───────────────────────────────────────────────
    /// Placeholder node representing a return value of a Call effectful op.
    /// The `u32` is a globally unique ID (from `FunctionBuilder::next_uid`);
    /// the `Type` is the type of this return value.
    ///
    /// Has no children. Cost is zero (instruction emitted by effectful lowering).
    CallResult(u32, Type),

    // ── Spill/reload pseudo-ops ──────────────────────────────────────────────
    /// GPR spill store: `operand[0]` is the VReg to spill, `i64` is the slot index.
    SpillStore(i64),
    /// GPR spill load: dst is the reload VReg, i64 is the slot index.
    SpillLoad(i64),
    /// XMM spill store: `operand[0]` is the VReg to spill, `i64` is the slot index.
    XmmSpillStore(i64),
    /// XMM spill load: dst is the reload VReg, i64 is the slot index.
    XmmSpillLoad(i64),

    /// Pseudo-instruction for Store barriers. Carries addr/val VRegs as operands
    /// so regalloc sees correct liveness. Skipped during lowering.
    StoreBarrier,

    /// Pseudo-instruction for void Call barriers (calls with no return value).
    /// Carries call-arg VRegs as operands so regalloc sees correct liveness.
    /// Skipped during lowering.
    VoidCallBarrier,

    /// Pseudo-instruction carrying the block terminator's arguments as operands.
    /// At most one per block, always its last instruction, skipped during
    /// lowering.
    ///
    /// The payload names which terminator argument each operand is: operand `j`
    /// carries argument `arg_indices[j]`. Arguments are numbered in a single
    /// sequence per terminator -- a Jump's args in order, a Branch's `true_args`
    /// followed by its `false_args`, a Ret's value as argument 0. The indices are
    /// explicit rather than implied by position because an argument can carry no
    /// operand at all: see [`PseudoOp::TerminatorArgs`] users for slot-routed
    /// parameters, which are passed in memory and hold no register here.
    ///
    /// This exists for the same reason the barrier pseudo-ops do, and closes the
    /// same hole one level further along. Everything else a block computes flows
    /// as a VReg that the splitter rewrites, coalescing renames and liveness
    /// sees. A terminator's arguments were `ClassId`s in the CFG that no operand
    /// rewrite reached, so three passes each re-derived them from
    /// `class_to_vreg` -- one for liveness, one for emission, and the splitter,
    /// which had no operand to act on at all and could only hope a recomputed
    /// use set happened to route through its reload segments. Whenever those
    /// derivations disagreed, a terminator read a register that no longer held
    /// the value.
    ///
    /// As a real operand list the splitter can see a terminator use, spill and
    /// reload against it, and both other passes read the answer instead of
    /// recomputing it.
    TerminatorArgs(Vec<u32>),
}

/// An operation, in whichever of the three worlds it belongs to.
///
/// The split is the point: a pure IR op cannot appear where a machine op is
/// required, and a pseudo that defines no value cannot reach a structure that
/// assumes one, without the type saying so.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    Pure(PureOp),
    Mach(MachOp),
    Pseudo(PseudoOp),
}

impl PseudoOp {
    /// Whether this pseudo takes no register, because it names no value.
    ///
    /// Only a pseudo can answer anything but `false`, which is why the predicate
    /// lives here rather than on `Op`: a pure or machine op is an expression and
    /// always defines something. Every bug in this area has been a structure that
    /// assumed otherwise -- a phantom `dst` taking a colour, a marker's position
    /// read as a def point.
    pub fn defines_no_value(&self) -> bool {
        matches!(
            self,
            PseudoOp::StoreBarrier
                | PseudoOp::VoidCallBarrier
                | PseudoOp::TerminatorArgs(_)
                // A spill store writes memory. Its consumers read the slot back
                // through a `SpillLoad`, never its `dst`.
                | PseudoOp::SpillStore(_)
                | PseudoOp::XmmSpillStore(_)
        )
    }
}

impl Op {
    /// Derive the result type of this node given the types of its children.
    ///
    /// Panics on type mismatches or wrong child counts.
    pub fn result_type(&self, child_types: &[Type]) -> Type {
        match self {
            // ── Arithmetic (binary, same integer type) ────────────────────────
            Op::Pure(PureOp::Add)
            | Op::Pure(PureOp::Sub)
            | Op::Pure(PureOp::Mul)
            | Op::Pure(PureOp::UDiv)
            | Op::Pure(PureOp::SDiv)
            | Op::Pure(PureOp::URem)
            | Op::Pure(PureOp::SRem) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                t.clone()
            }

            // ── Bitwise (binary, same integer type) ──────────────────────────
            Op::Pure(PureOp::And) | Op::Pure(PureOp::Or) | Op::Pure(PureOp::Xor) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                t.clone()
            }

            // ── Shifts (two integer operands, may differ; result = first) ────
            Op::Pure(PureOp::Shl) | Op::Pure(PureOp::Shr) | Op::Pure(PureOp::Sar) => {
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
            Op::Pure(PureOp::Sext(target)) => {
                assert_eq!(child_types.len(), 1, "Sext requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "Sext requires integer child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Pure(PureOp::Zext(target)) => {
                assert_eq!(child_types.len(), 1, "Zext requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "Zext requires integer child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Pure(PureOp::Trunc(target)) => {
                assert_eq!(child_types.len(), 1, "Trunc requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "Trunc requires integer child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Pure(PureOp::Bitcast(target)) => {
                assert_eq!(child_types.len(), 1, "Bitcast requires 1 child");
                target.clone()
            }

            // ── Constants (0 children) ────────────────────────────────────────
            Op::Pure(PureOp::Iconst(_val, ty)) => {
                assert_eq!(child_types.len(), 0, "Iconst requires 0 children");
                ty.clone()
            }
            Op::Pure(PureOp::Fconst(_, ty)) => {
                assert_eq!(child_types.len(), 0, "Fconst requires 0 children");
                ty.clone()
            }
            Op::Pure(PureOp::Param(_idx, ty)) => {
                assert_eq!(child_types.len(), 0, "Param requires 0 children");
                ty.clone()
            }
            Op::Pure(PureOp::BlockParam(_block_id, _param_idx, ty)) => {
                assert_eq!(child_types.len(), 0, "BlockParam requires 0 children");
                ty.clone()
            }
            Op::Pseudo(PseudoOp::LoadResult(_uid, ty)) => {
                assert_eq!(child_types.len(), 0, "LoadResult requires 0 children");
                ty.clone()
            }
            Op::Pseudo(PseudoOp::CallResult(_idx, ty)) => {
                assert_eq!(child_types.len(), 0, "CallResult requires 0 children");
                ty.clone()
            }
            Op::Pseudo(PseudoOp::StackAddr(_)) => {
                assert_eq!(child_types.len(), 0, "StackAddr requires 0 children");
                Type::I64
            }
            Op::Pseudo(PseudoOp::GlobalAddr(_)) => {
                assert_eq!(child_types.len(), 0, "GlobalAddr requires 0 children");
                Type::I64
            }

            // ── Comparison ────────────────────────────────────────────────────
            Op::Pure(PureOp::Icmp(_cc)) => {
                assert_eq!(child_types.len(), 2, "Icmp requires 2 children");
                let t = &child_types[0];
                assert!(t.is_integer(), "Icmp requires integer operands, got {t:?}");
                assert_eq!(
                    &child_types[1], t,
                    "Icmp operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                Type::Flags
            }
            Op::Pure(PureOp::Fcmp(_cc)) => {
                assert_eq!(child_types.len(), 2, "Fcmp requires 2 children");
                let t = &child_types[0];
                assert!(t.is_float(), "Fcmp requires float operands, got {t:?}");
                assert_eq!(
                    &child_types[1], t,
                    "Fcmp operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                Type::Flags
            }

            // ── Float/int conversions ────────────────────────────────────────
            Op::Pure(PureOp::IntToFloat(target)) => {
                assert_eq!(child_types.len(), 1, "IntToFloat requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "IntToFloat requires integer child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Pure(PureOp::FloatToInt(target)) => {
                assert_eq!(child_types.len(), 1, "FloatToInt requires 1 child");
                assert!(
                    child_types[0].is_float(),
                    "FloatToInt requires float child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Pure(PureOp::FloatExt) => {
                assert_eq!(child_types.len(), 1, "FloatExt requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F32,
                    "FloatExt requires F32 child, got {:?}",
                    child_types[0]
                );
                Type::F64
            }
            Op::Pure(PureOp::FloatTrunc) => {
                assert_eq!(child_types.len(), 1, "FloatTrunc requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F64,
                    "FloatTrunc requires F64 child, got {:?}",
                    child_types[0]
                );
                Type::F32
            }

            // ── FP binary ops ─────────────────────────────────────────────────
            Op::Pure(PureOp::Fadd)
            | Op::Pure(PureOp::Fsub)
            | Op::Pure(PureOp::Fmul)
            | Op::Pure(PureOp::Fdiv) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert!(
                    child_types[0].is_float(),
                    "{self:?} requires float operands, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[0], child_types[1],
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    child_types[0], child_types[1]
                );
                child_types[0].clone()
            }
            Op::Pure(PureOp::Fsqrt) => {
                assert_eq!(child_types.len(), 1, "Fsqrt requires 1 child");
                assert!(
                    child_types[0].is_float(),
                    "Fsqrt requires float operand, got {:?}",
                    child_types[0]
                );
                child_types[0].clone()
            }

            // ── Select ────────────────────────────────────────────────────────
            Op::Pure(PureOp::Select(_)) => {
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
            Op::Pure(PureOp::Proj0) => {
                assert_eq!(child_types.len(), 1, "Proj0 requires 1 child");
                match &child_types[0] {
                    Type::Pair(a, _b) => *a.clone(),
                    other => panic!("Proj0 requires Pair child, got {other:?}"),
                }
            }
            Op::Pure(PureOp::Proj1) => {
                assert_eq!(child_types.len(), 1, "Proj1 requires 1 child");
                match &child_types[0] {
                    Type::Pair(_a, b) => *b.clone(),
                    other => panic!("Proj1 requires Pair child, got {other:?}"),
                }
            }

            // ── x86 ALU (binary integer → Pair(childtype, Flags)) ────────────
            Op::Mach(MachOp::X86Add)
            | Op::Mach(MachOp::X86Sub)
            | Op::Mach(MachOp::X86And)
            | Op::Mach(MachOp::X86Or)
            | Op::Mach(MachOp::X86Xor) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                Type::Pair(Box::new(t.clone()), Box::new(Type::Flags))
            }
            Op::Mach(MachOp::X86Shl)
            | Op::Mach(MachOp::X86Sar)
            | Op::Mach(MachOp::X86Shr)
            | Op::Mach(MachOp::X86Bts)
            | Op::Mach(MachOp::X86Btr)
            | Op::Mach(MachOp::X86Btc) => {
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
            Op::Mach(MachOp::X86ShldImm(_)) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                Type::Pair(Box::new(t.clone()), Box::new(Type::Flags))
            }

            // ── x86 immediate-form ALU and shifts (1 child → Pair(childtype, Flags)) ──
            Op::Mach(MachOp::X86AddI(_))
            | Op::Mach(MachOp::X86SubI(_))
            | Op::Mach(MachOp::X86ImulI(_))
            | Op::Mach(MachOp::X86AndI(_))
            | Op::Mach(MachOp::X86OrI(_))
            | Op::Mach(MachOp::X86XorI(_))
            | Op::Mach(MachOp::X86ShlImm(_))
            | Op::Mach(MachOp::X86ShrImm(_))
            | Op::Mach(MachOp::X86SarImm(_))
            | Op::Mach(MachOp::X86RolImm(_))
            | Op::Mach(MachOp::X86BtsI(_))
            | Op::Mach(MachOp::X86BtrI(_))
            | Op::Mach(MachOp::X86BtcI(_)) => {
                assert_eq!(child_types.len(), 1, "{self:?} requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "{self:?} operand must be integer, got {:?}",
                    child_types[0]
                );
                Type::Pair(Box::new(child_types[0].clone()), Box::new(Type::Flags))
            }

            // ── x86 flag-only compare with immediate (1 child → Flags) ───────────
            Op::Mach(MachOp::X86CmpI { .. }) => {
                assert_eq!(child_types.len(), 1, "X86CmpI requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "X86CmpI operand must be integer, got {:?}",
                    child_types[0]
                );
                Type::Flags
            }

            // ── x86 LEA variants (I64, I64 → I64) ───────────────────────────
            Op::Mach(MachOp::X86Lea2) | Op::Mach(MachOp::X86Lea3 { .. }) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert!(
                    matches!(child_types[0], Type::I32 | Type::I64),
                    "{self:?} requires I32 or I64 base, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[0], child_types[1],
                    "{self:?} base and index must have same type"
                );
                child_types[0].clone()
            }
            Op::Mach(MachOp::X86Lea4 { .. }) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert!(
                    matches!(child_types[0], Type::I32 | Type::I64),
                    "{self:?} requires I32 or I64 base, got {:?}",
                    child_types[0]
                );
                // child_types[1] may be NONE (no index register), skip type check
                child_types[0].clone()
            }

            // ── X86Idiv / X86Div (2 integer children → Pair(I64, I64)) ────────
            // Proj0 = quotient (RAX), Proj1 = remainder (RDX).
            Op::Mach(MachOp::X86Idiv(ty)) | Op::Mach(MachOp::X86Div(ty)) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                let t = &child_types[0];
                assert_eq!(
                    t, ty,
                    "{self:?} operand type disagrees with the type on the op"
                );
                assert!(
                    t.is_integer(),
                    "{self:?} requires integer operands, got {t:?}"
                );
                assert_eq!(
                    &child_types[1], t,
                    "{self:?} operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                Type::Pair(Box::new(t.clone()), Box::new(t.clone()))
            }

            // ── X86Imul3 (2 children → Pair(childtype, Flags)) ──────────────
            Op::Mach(MachOp::X86Imul3) => {
                assert_eq!(child_types.len(), 2, "X86Imul3 requires 2 children");
                let t = &child_types[0];
                assert!(
                    t.is_integer(),
                    "X86Imul3 first operand must be integer, got {:?}",
                    t
                );
                assert_eq!(
                    &child_types[1], t,
                    "X86Imul3 operand type mismatch: {:?} vs {:?}",
                    t, child_types[1]
                );
                Type::Pair(Box::new(t.clone()), Box::new(Type::Flags))
            }

            // ── X86Cmov (flags, t, f → t's type) ────────────────────────────
            Op::Mach(MachOp::X86Cmov(_cc)) => {
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
            Op::Mach(MachOp::X86Setcc(_cc)) => {
                assert_eq!(child_types.len(), 1, "X86Setcc requires 1 child");
                assert_eq!(child_types[0], Type::Flags, "X86Setcc child must be Flags");
                Type::I8
            }

            // ── X86SbbSelf (flags → the width it carries) ─────────────────────
            Op::Mach(MachOp::X86SbbSelf(ty)) => {
                assert_eq!(child_types.len(), 1, "X86SbbSelf requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::Flags,
                    "X86SbbSelf child must be Flags"
                );
                ty.clone()
            }

            // ── x86 FP binary ops (F64, F64 → F64) ──────────────────────────
            Op::Mach(MachOp::X86Addsd)
            | Op::Mach(MachOp::X86Subsd)
            | Op::Mach(MachOp::X86Mulsd)
            | Op::Mach(MachOp::X86Divsd) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert_eq!(
                    child_types[0],
                    Type::F64,
                    "{self:?} requires F64 operands, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[1],
                    Type::F64,
                    "{self:?} requires F64 operands, got {:?}",
                    child_types[1]
                );
                Type::F64
            }
            Op::Mach(MachOp::X86Sqrtsd) => {
                assert_eq!(child_types.len(), 1, "X86Sqrtsd requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F64,
                    "X86Sqrtsd requires F64 operand, got {:?}",
                    child_types[0]
                );
                Type::F64
            }

            // ── x86 FP binary ops (F32, F32 → F32) ──────────────────────────
            Op::Mach(MachOp::X86Addss)
            | Op::Mach(MachOp::X86Subss)
            | Op::Mach(MachOp::X86Mulss)
            | Op::Mach(MachOp::X86Divss) => {
                assert_eq!(child_types.len(), 2, "{self:?} requires 2 children");
                assert_eq!(
                    child_types[0],
                    Type::F32,
                    "{self:?} requires F32 operands, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[1],
                    Type::F32,
                    "{self:?} requires F32 operands, got {:?}",
                    child_types[1]
                );
                Type::F32
            }
            Op::Mach(MachOp::X86Sqrtss) => {
                assert_eq!(child_types.len(), 1, "X86Sqrtss requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F32,
                    "X86Sqrtss requires F32 operand, got {:?}",
                    child_types[0]
                );
                Type::F32
            }

            // ── x86 FP conversion ops ────────────────────────────────────────
            Op::Mach(MachOp::X86Cvtsi2sd(src_ty)) => {
                assert_eq!(child_types.len(), 1, "X86Cvtsi2sd requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "X86Cvtsi2sd requires integer child, got {:?}",
                    child_types[0]
                );
                assert!(
                    matches!(src_ty, Type::I32 | Type::I64),
                    "X86Cvtsi2sd source must be I32 or I64, got {src_ty:?}"
                );
                Type::F64
            }
            Op::Mach(MachOp::X86Cvtsi2ss(src_ty)) => {
                assert_eq!(child_types.len(), 1, "X86Cvtsi2ss requires 1 child");
                assert!(
                    child_types[0].is_integer(),
                    "X86Cvtsi2ss requires integer child, got {:?}",
                    child_types[0]
                );
                assert!(
                    matches!(src_ty, Type::I32 | Type::I64),
                    "X86Cvtsi2ss source must be I32 or I64, got {src_ty:?}"
                );
                Type::F32
            }
            Op::Mach(MachOp::X86Cvttsd2si(target)) => {
                assert_eq!(child_types.len(), 1, "X86Cvttsd2si requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F64,
                    "X86Cvttsd2si requires F64 child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Mach(MachOp::X86Cvttss2si(target)) => {
                assert_eq!(child_types.len(), 1, "X86Cvttss2si requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F32,
                    "X86Cvttss2si requires F32 child, got {:?}",
                    child_types[0]
                );
                target.clone()
            }
            Op::Mach(MachOp::X86Cvtsd2ss) => {
                assert_eq!(child_types.len(), 1, "X86Cvtsd2ss requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F64,
                    "X86Cvtsd2ss requires F64 child, got {:?}",
                    child_types[0]
                );
                Type::F32
            }
            Op::Mach(MachOp::X86Cvtss2sd) => {
                assert_eq!(child_types.len(), 1, "X86Cvtss2sd requires 1 child");
                assert_eq!(
                    child_types[0],
                    Type::F32,
                    "X86Cvtss2sd requires F32 child, got {:?}",
                    child_types[0]
                );
                Type::F64
            }

            // ── x86 FP comparison ops ────────────────────────────────────────
            Op::Mach(MachOp::X86Ucomisd) => {
                assert_eq!(child_types.len(), 2, "X86Ucomisd requires 2 children");
                assert_eq!(
                    child_types[0],
                    Type::F64,
                    "X86Ucomisd requires F64 operands, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[1],
                    Type::F64,
                    "X86Ucomisd requires F64 operands, got {:?}",
                    child_types[1]
                );
                Type::Flags
            }
            Op::Mach(MachOp::X86UcomisdCc(_)) => {
                assert_eq!(child_types.len(), 2, "X86UcomisdCc requires 2 children");
                assert_eq!(
                    child_types[0],
                    Type::F64,
                    "X86UcomisdCc requires F64 operands, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[1],
                    Type::F64,
                    "X86UcomisdCc requires F64 operands, got {:?}",
                    child_types[1]
                );
                Type::Flags
            }
            Op::Mach(MachOp::X86UcomissCc(_)) => {
                assert_eq!(child_types.len(), 2, "X86UcomissCc requires 2 children");
                assert_eq!(
                    child_types[0],
                    Type::F32,
                    "X86UcomissCc requires F32 operands, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[1],
                    Type::F32,
                    "X86UcomissCc requires F32 operands, got {:?}",
                    child_types[1]
                );
                Type::Flags
            }
            Op::Mach(MachOp::X86Ucomiss) => {
                assert_eq!(child_types.len(), 2, "X86Ucomiss requires 2 children");
                assert_eq!(
                    child_types[0],
                    Type::F32,
                    "X86Ucomiss requires F32 operands, got {:?}",
                    child_types[0]
                );
                assert_eq!(
                    child_types[1],
                    Type::F32,
                    "X86Ucomiss requires F32 operands, got {:?}",
                    child_types[1]
                );
                Type::Flags
            }

            // ── Addr (base I64, index I64 → I64) ─────────────────────────────
            Op::Pure(PureOp::Addr { .. }) => {
                assert_eq!(
                    child_types.len(),
                    2,
                    "Addr requires 2 children (base, index)"
                );
                assert_eq!(child_types[0], Type::I64, "Addr base must be I64");
                assert_eq!(child_types[1], Type::I64, "Addr index must be I64");
                Type::I64
            }

            // ── x86-64 conversion ops (1 child → to type) ────────────────────
            Op::Mach(MachOp::X86Movsx { from, to }) => {
                assert_eq!(child_types.len(), 1, "X86Movsx requires 1 child");
                assert_eq!(
                    &child_types[0], from,
                    "X86Movsx child type mismatch: expected {from:?}, got {:?}",
                    child_types[0]
                );
                to.clone()
            }
            Op::Mach(MachOp::X86Movzx { from, to }) => {
                assert_eq!(child_types.len(), 1, "X86Movzx requires 1 child");
                assert_eq!(
                    &child_types[0], from,
                    "X86Movzx child type mismatch: expected {from:?}, got {:?}",
                    child_types[0]
                );
                to.clone()
            }
            Op::Mach(MachOp::X86Trunc { from, to }) => {
                assert_eq!(child_types.len(), 1, "X86Trunc requires 1 child");
                assert_eq!(
                    &child_types[0], from,
                    "X86Trunc child type mismatch: expected {from:?}, got {:?}",
                    child_types[0]
                );
                to.clone()
            }
            Op::Mach(MachOp::X86Bitcast { from, to }) => {
                assert_eq!(child_types.len(), 1, "X86Bitcast requires 1 child");
                assert_eq!(
                    &child_types[0], from,
                    "X86Bitcast child type mismatch: expected {from:?}, got {:?}",
                    child_types[0]
                );
                to.clone()
            }

            // Spill pseudo-ops are never type-checked via result_type; they are
            // internal markers consumed by the lowering pass.
            Op::Pseudo(PseudoOp::SpillStore(_))
            | Op::Pseudo(PseudoOp::SpillLoad(_))
            | Op::Pseudo(PseudoOp::XmmSpillStore(_))
            | Op::Pseudo(PseudoOp::XmmSpillLoad(_)) => {
                unreachable!("spill pseudo-ops have no result_type")
            }

            Op::Pseudo(PseudoOp::StoreBarrier)
            | Op::Pseudo(PseudoOp::VoidCallBarrier)
            | Op::Pseudo(PseudoOp::TerminatorArgs(_)) => Type::I64,
        }
    }

    /// Whether this op only *names* a value something else already placed.
    ///
    /// A block parameter is written by the phi copies on the edge and a
    /// function parameter by the caller, so both hold their register before the
    /// block's first instruction runs. The marker says where the value is
    /// named, and the scheduler puts it wherever the dependence order allows --
    /// so the value is live from block entry to its marker and interferes with
    /// everything the block names over that run, which a backward liveness scan
    /// starting at the marker does not see.
    pub fn is_param_marker(&self) -> bool {
        matches!(
            self,
            Op::Pure(PureOp::BlockParam(..)) | Op::Pure(PureOp::Param(..))
        )
    }

    /// Whether this op defines nothing, so its `dst` names no value.
    ///
    /// The barrier pseudo-ops exist to carry operands into liveness and are
    /// skipped during lowering; nothing ever reads what they "define". Their
    /// `dst` must therefore take no interference edges and count for no
    /// pressure. Left in, it is a value the allocator has to colour at the
    /// widest point in the block -- a terminator's argument list is the whole
    /// parallel copy, so the phantom is the difference between fitting the
    /// budget and overshooting it by one.
    pub fn has_no_result(&self) -> bool {
        match self {
            Op::Pseudo(p) => p.defines_no_value(),
            // A pure or machine op is an expression: it always names a value.
            Op::Pure(_) | Op::Mach(_) => false,
        }
    }

    /// Whether this op leaves its result in fixed physical registers rather than
    /// in the VReg it defines.
    ///
    /// A division writes its quotient to RAX and its remainder to RDX. The pair
    /// VReg names the two together and holds neither, and only the projections
    /// adjacent to the division read those registers -- so it needs no colour of
    /// its own, and a slot saving it would store a register that never held it.
    /// Spilling it is worse than wasteful: the reload renames the operand the
    /// projections are recognised by, and a projection that is no longer read as
    /// a division's becomes an ordinary register copy out of the pair.
    pub fn result_in_fixed_regs(&self) -> bool {
        matches!(self, Op::Mach(MachOp::X86Idiv(..) | MachOp::X86Div(..)))
    }

    /// The operand index this op's result must share a register with, if any.
    ///
    /// x86 ALU and SSE arithmetic is two-address: `add dst, src` reads and
    /// writes `dst`. `lower.rs` honours that by emitting `mov dst, operand[0]`
    /// whenever the allocator did not give the result its first operand's
    /// register, so every op named here costs a copy unless the two are
    /// coloured alike.
    ///
    /// The allocator reads this to bias its colouring toward the register the
    /// result wants. That is why the answer lives on the op rather than in
    /// either pass: lowering and the allocator disagreeing about which operand
    /// is destructive means a copy nobody planned, or a copy planned and never
    /// emitted.
    pub fn two_address_src(&self) -> Option<usize> {
        matches!(
            self,
            Op::Mach(
                MachOp::X86Add
                    | MachOp::X86Sub
                    | MachOp::X86And
                    | MachOp::X86Or
                    | MachOp::X86Xor
                    | MachOp::X86AddI(_)
                    | MachOp::X86SubI(_)
                    | MachOp::X86AndI(_)
                    | MachOp::X86OrI(_)
                    | MachOp::X86XorI(_)
                    | MachOp::X86Shl
                    | MachOp::X86Shr
                    | MachOp::X86Sar
                    | MachOp::X86ShlImm(_)
                    | MachOp::X86ShrImm(_)
                    | MachOp::X86SarImm(_)
                    | MachOp::X86RolImm(_)
                    | MachOp::X86ShldImm(_)
                    | MachOp::X86Bts
                    | MachOp::X86Btr
                    | MachOp::X86Btc
                    | MachOp::X86BtsI(_)
                    | MachOp::X86BtrI(_)
                    | MachOp::X86BtcI(_)
                    | MachOp::X86Addsd
                    | MachOp::X86Subsd
                    | MachOp::X86Mulsd
                    | MachOp::X86Divsd
                    | MachOp::X86Addss
                    | MachOp::X86Subss
                    | MachOp::X86Mulss
                    | MachOp::X86Divss
            )
        )
        .then_some(0)
    }

    /// Whether this op's result *is* the flags, rather than a value in a
    /// register.
    ///
    /// The ops whose `result_type` is `Type::Flags`. A pair-producing op whose
    /// second element is flags is not one of these -- the pair is a real value
    /// and it is the `Proj1` that names the flags.
    pub fn produces_flags(&self) -> bool {
        matches!(
            self,
            Op::Pure(PureOp::Icmp(_))
                | Op::Pure(PureOp::Fcmp(_))
                | Op::Mach(MachOp::X86CmpI { .. })
                | Op::Mach(MachOp::X86Ucomisd)
                | Op::Mach(MachOp::X86Ucomiss)
                | Op::Mach(MachOp::X86UcomisdCc(_))
                | Op::Mach(MachOp::X86UcomissCc(_))
        )
    }

    /// Returns true if this op produces a value that lives in an XMM (FP) register.
    pub fn is_fp_op(&self) -> bool {
        match self {
            // F64 arithmetic
            Op::Mach(MachOp::X86Addsd)
            | Op::Mach(MachOp::X86Subsd)
            | Op::Mach(MachOp::X86Mulsd)
            | Op::Mach(MachOp::X86Divsd)
            | Op::Mach(MachOp::X86Sqrtsd)
            // F32 arithmetic
            | Op::Mach(MachOp::X86Addss)
            | Op::Mach(MachOp::X86Subss)
            | Op::Mach(MachOp::X86Mulss)
            | Op::Mach(MachOp::X86Divss)
            | Op::Mach(MachOp::X86Sqrtss)
            // Conversions that produce XMM results
            | Op::Mach(MachOp::X86Cvtsi2sd(_))
            | Op::Mach(MachOp::X86Cvtsi2ss(_))
            | Op::Mach(MachOp::X86Cvtsd2ss)
            | Op::Mach(MachOp::X86Cvtss2sd)
            // FP constants
            | Op::Pure(PureOp::Fconst(_, _))
            // XMM spill reloads produce XMM values
            | Op::Pseudo(PseudoOp::XmmSpillLoad(_)) => true,
            // Block parameters (phi destinations) with float types
            Op::Pure(PureOp::BlockParam(_, _, ty)) => ty.is_float(),
            // Call results with float return types
            Op::Pseudo(PseudoOp::CallResult(_, ty)) => ty.is_float(),
            // Load results with float types
            Op::Pseudo(PseudoOp::LoadResult(_, ty)) => ty.is_float(),
            // Function parameters with float types
            Op::Pure(PureOp::Param(_, ty)) => ty.is_float(),
            Op::Mach(MachOp::X86Bitcast { to, .. }) => matches!(to, Type::F32 | Type::F64),
            // X86Cvttsd2si / X86Cvttss2si produce GPR (not XMM)
            // X86Ucomisd / X86Ucomiss produce flags (not XMM)
            _ => false,
        }
    }

    /// Returns true if this op reads its operands from a different register
    /// class than the one it writes.
    ///
    /// x86-64 has exactly two of these shapes and both are conversions:
    /// `cvtsi2sd/ss` and a GPR-to-XMM `movq` write XMM from a GPR, while
    /// `cvttsd2si/cvttss2si`, `ucomisd/ucomiss` and an XMM-to-GPR `movq` read
    /// XMM and write a GPR or flags. Every other op keeps both sides in one
    /// class.
    ///
    /// Inferring operand class from [`Self::is_fp_op`] (which describes the
    /// *result*) puts an integer value in an XMM register for these ops.
    pub fn has_cross_class_operands(&self) -> bool {
        matches!(
            self,
            Op::Mach(MachOp::X86Cvtsi2sd(_))
                | Op::Mach(MachOp::X86Cvtsi2ss(_))
                | Op::Mach(MachOp::X86Cvttsd2si(_))
                | Op::Mach(MachOp::X86Cvttss2si(_))
                | Op::Mach(MachOp::X86Ucomisd)
                | Op::Mach(MachOp::X86Ucomiss)
                | Op::Mach(MachOp::X86UcomisdCc(_))
                | Op::Mach(MachOp::X86UcomissCc(_))
                | Op::Mach(MachOp::X86Bitcast { .. })
        )
    }

    /// The register class this op reads its operands from.
    ///
    /// For the cross-class ops above this is the opposite of the result class;
    /// for everything else it matches.
    pub fn operand_reg_class(&self) -> RegClass {
        match self {
            // XMM result, GPR source.
            Op::Mach(MachOp::X86Cvtsi2sd(_)) | Op::Mach(MachOp::X86Cvtsi2ss(_)) => RegClass::GPR,
            // GPR or flags result, XMM source.
            Op::Mach(MachOp::X86Cvttsd2si(_))
            | Op::Mach(MachOp::X86Cvttss2si(_))
            | Op::Mach(MachOp::X86Ucomisd)
            | Op::Mach(MachOp::X86Ucomiss)
            | Op::Mach(MachOp::X86UcomisdCc(_))
            | Op::Mach(MachOp::X86UcomissCc(_)) => RegClass::XMM,
            // movq between the classes: the source is whichever side `from` is.
            Op::Mach(MachOp::X86Bitcast { from, .. }) => {
                if from.is_float() {
                    RegClass::XMM
                } else {
                    RegClass::GPR
                }
            }
            // A load's operands are its address -- the folded `Addr`, and the
            // base and index the barrier repeats for liveness -- so they are
            // GPR however the loaded value is classed. This is the one op whose
            // result is FP while no operand of it is.
            Op::Pseudo(PseudoOp::LoadResult(_, _)) => RegClass::GPR,
            _ if self.is_fp_op() => RegClass::XMM,
            _ => RegClass::GPR,
        }
    }

    /// Returns true if this op can be cheaply recomputed instead of spilled.
    ///
    /// These ops take no operands and recompute their value from nothing, so a
    /// copy of one is as good as the original wherever it is placed. `Fconst`
    /// costs two instructions (`movabs` into the scratch GPR, then `movq` into
    /// an XMM) and still qualifies.
    ///
    /// Cheapness is NOT the test, which is the mistake this predicate exists to
    /// prevent. `BlockParam`, `Param`, `CallResult` and `LoadResult` are leaves
    /// of cost ~0, but their value is already in a register at one particular
    /// point -- put there by a predecessor's phi copy, by the caller, or by the
    /// instruction before -- and re-emitting the pseudo-op mints a VReg that no
    /// instruction ever writes.
    pub fn is_rematerializable(&self) -> bool {
        matches!(
            self,
            Op::Pure(PureOp::Iconst(_, _))
                | Op::Pure(PureOp::Fconst(_, _))
                | Op::Pseudo(PseudoOp::StackAddr(_))
                | Op::Pseudo(PseudoOp::GlobalAddr(_))
        )
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
        let ty = Op::Pure(PureOp::Add).result_type(&[Type::I32, Type::I32]);
        assert_eq!(ty, Type::I32);
    }

    #[test]
    fn add_i64() {
        let ty = Op::Pure(PureOp::Add).result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    #[should_panic]
    fn add_type_mismatch() {
        Op::Pure(PureOp::Add).result_type(&[Type::I32, Type::I64]);
    }

    #[test]
    #[should_panic]
    fn add_float_rejected() {
        Op::Pure(PureOp::Add).result_type(&[Type::F64, Type::F64]);
    }

    #[test]
    fn sub_i64() {
        assert_eq!(
            Op::Pure(PureOp::Sub).result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    fn mul_i16() {
        assert_eq!(
            Op::Pure(PureOp::Mul).result_type(&[Type::I16, Type::I16]),
            Type::I16
        );
    }

    #[test]
    fn udiv_i32() {
        assert_eq!(
            Op::Pure(PureOp::UDiv).result_type(&[Type::I32, Type::I32]),
            Type::I32
        );
    }

    #[test]
    fn urem_i8() {
        assert_eq!(
            Op::Pure(PureOp::URem).result_type(&[Type::I8, Type::I8]),
            Type::I8
        );
    }

    // ── Bitwise ───────────────────────────────────────────────────────────────

    #[test]
    fn and_i64() {
        assert_eq!(
            Op::Pure(PureOp::And).result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    fn xor_i32() {
        assert_eq!(
            Op::Pure(PureOp::Xor).result_type(&[Type::I32, Type::I32]),
            Type::I32
        );
    }

    #[test]
    fn shl_different_widths() {
        // shift amount can differ from value type
        let ty = Op::Pure(PureOp::Shl).result_type(&[Type::I64, Type::I8]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    fn sar_i32() {
        assert_eq!(
            Op::Pure(PureOp::Sar).result_type(&[Type::I32, Type::I32]),
            Type::I32
        );
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    #[test]
    fn sext_i32_to_i64() {
        assert_eq!(
            Op::Pure(PureOp::Sext(Type::I64)).result_type(&[Type::I32]),
            Type::I64
        );
    }

    #[test]
    fn zext_i8_to_i64() {
        assert_eq!(
            Op::Pure(PureOp::Zext(Type::I64)).result_type(&[Type::I8]),
            Type::I64
        );
    }

    #[test]
    fn trunc_i64_to_i32() {
        assert_eq!(
            Op::Pure(PureOp::Trunc(Type::I32)).result_type(&[Type::I64]),
            Type::I32
        );
    }

    #[test]
    fn bitcast_i64_to_f64() {
        assert_eq!(
            Op::Pure(PureOp::Bitcast(Type::F64)).result_type(&[Type::I64]),
            Type::F64
        );
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    #[test]
    fn iconst_i64() {
        assert_eq!(
            Op::Pure(PureOp::Iconst(42, Type::I64)).result_type(&[]),
            Type::I64
        );
    }

    #[test]
    fn iconst_i32() {
        assert_eq!(
            Op::Pure(PureOp::Iconst(0, Type::I32)).result_type(&[]),
            Type::I32
        );
    }

    #[test]
    fn fconst_is_f64() {
        assert_eq!(
            Op::Pure(PureOp::Fconst(0u64, Type::F64)).result_type(&[]),
            Type::F64
        );
    }

    #[test]
    fn fconst_is_f32() {
        assert_eq!(
            Op::Pure(PureOp::Fconst(0u64, Type::F32)).result_type(&[]),
            Type::F32
        );
    }

    // ── Comparison ────────────────────────────────────────────────────────────

    #[test]
    fn icmp_produces_flags() {
        assert_eq!(
            Op::Pure(PureOp::Icmp(CondCode::Slt)).result_type(&[Type::I64, Type::I64]),
            Type::Flags
        );
    }

    #[test]
    fn icmp_eq_i32() {
        assert_eq!(
            Op::Pure(PureOp::Icmp(CondCode::Eq)).result_type(&[Type::I32, Type::I32]),
            Type::Flags
        );
    }

    #[test]
    #[should_panic]
    fn icmp_type_mismatch() {
        Op::Pure(PureOp::Icmp(CondCode::Eq)).result_type(&[Type::I32, Type::I64]);
    }

    // ── FP ops ────────────────────────────────────────────────────────────────

    #[test]
    fn fadd_f64() {
        assert_eq!(
            Op::Pure(PureOp::Fadd).result_type(&[Type::F64, Type::F64]),
            Type::F64
        );
    }

    #[test]
    fn fsqrt_f64() {
        assert_eq!(Op::Pure(PureOp::Fsqrt).result_type(&[Type::F64]), Type::F64);
    }

    #[test]
    #[should_panic]
    fn fadd_wrong_type() {
        Op::Pure(PureOp::Fadd).result_type(&[Type::I64, Type::I64]);
    }

    // ── Select ────────────────────────────────────────────────────────────────

    #[test]
    fn select_i64() {
        let ty = Op::Pure(PureOp::Select(CondCode::Ne)).result_type(&[
            Type::Flags,
            Type::I64,
            Type::I64,
        ]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    #[should_panic]
    fn select_branch_mismatch() {
        Op::Pure(PureOp::Select(CondCode::Ne)).result_type(&[Type::Flags, Type::I32, Type::I64]);
    }

    // ── Projections ───────────────────────────────────────────────────────────

    #[test]
    fn proj0_pair() {
        let pair = Type::Pair(Box::new(Type::I64), Box::new(Type::Flags));
        assert_eq!(Op::Pure(PureOp::Proj0).result_type(&[pair]), Type::I64);
    }

    #[test]
    fn proj1_pair() {
        let pair = Type::Pair(Box::new(Type::I64), Box::new(Type::Flags));
        assert_eq!(Op::Pure(PureOp::Proj1).result_type(&[pair]), Type::Flags);
    }

    #[test]
    #[should_panic]
    fn proj0_non_pair() {
        Op::Pure(PureOp::Proj0).result_type(&[Type::I64]);
    }

    // ── x86-64 machine ops ────────────────────────────────────────────────────

    #[test]
    fn x86add_produces_pair() {
        let ty = Op::Mach(MachOp::X86Add).result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86sub_produces_pair() {
        let ty = Op::Mach(MachOp::X86Sub).result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86and_i32() {
        let ty = Op::Mach(MachOp::X86And).result_type(&[Type::I32, Type::I32]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I32), Box::new(Type::Flags)));
    }

    #[test]
    fn x86shl_produces_pair() {
        let ty = Op::Mach(MachOp::X86Shl).result_type(&[Type::I64, Type::I8]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86lea2_i64() {
        assert_eq!(
            Op::Mach(MachOp::X86Lea2).result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    fn x86lea3_i64() {
        assert_eq!(
            Op::Mach(MachOp::X86Lea3 { scale: 2 }).result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    fn x86lea4_i64() {
        assert_eq!(
            Op::Mach(MachOp::X86Lea4 { scale: 4, disp: 16 }).result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    fn x86imul3_pair() {
        let ty = Op::Mach(MachOp::X86Imul3).result_type(&[Type::I64, Type::I64]);
        assert_eq!(ty, Type::Pair(Box::new(Type::I64), Box::new(Type::Flags)));
    }

    #[test]
    fn x86cmov_i64() {
        let ty = Op::Mach(MachOp::X86Cmov(CondCode::Ne)).result_type(&[
            Type::Flags,
            Type::I64,
            Type::I64,
        ]);
        assert_eq!(ty, Type::I64);
    }

    #[test]
    fn x86setcc_i8() {
        assert_eq!(
            Op::Mach(MachOp::X86Setcc(CondCode::Eq)).result_type(&[Type::Flags]),
            Type::I8
        );
    }

    #[test]
    fn addr_i64() {
        assert_eq!(
            Op::Pure(PureOp::Addr { scale: 4, disp: 8 }).result_type(&[Type::I64, Type::I64]),
            Type::I64
        );
    }

    #[test]
    #[should_panic]
    fn x86add_flags_rejected() {
        Op::Mach(MachOp::X86Add).result_type(&[Type::Flags, Type::Flags]);
    }

    #[test]
    #[should_panic]
    fn x86cmov_wrong_flags() {
        Op::Mach(MachOp::X86Cmov(CondCode::Eq)).result_type(&[Type::I64, Type::I64, Type::I64]);
    }

    // ── ClassId sentinel ──────────────────────────────────────────────────────

    #[test]
    fn classid_none_is_max() {
        assert_eq!(ClassId::NONE, ClassId(u32::MAX));
    }
}
