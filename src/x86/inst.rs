use crate::egraph::extract::VReg;
use crate::ir::condcode::CondCode;
use crate::ir::types::Type;

use super::addr::Addr;
use super::reg::Reg;

pub type LabelId = u32;
pub type Symbol = String;

/// Operand size for width-dependent x86-64 instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpSize {
    S8,
    S16,
    S32,
    S64,
}

impl OpSize {
    pub fn from_type(ty: &Type) -> Self {
        match ty {
            Type::I8 => OpSize::S8,
            Type::I16 => OpSize::S16,
            Type::I32 => OpSize::S32,
            Type::I64 => OpSize::S64,
            _ => panic!("OpSize::from_type: unsupported type {ty:?}"),
        }
    }

    pub fn byte_width(self) -> u32 {
        match self {
            OpSize::S8 => 1,
            OpSize::S16 => 2,
            OpSize::S32 => 4,
            OpSize::S64 => 8,
        }
    }
}

/// A physical register or virtual register operand.
///
/// Before register allocation operands may be `VReg`. After allocation they
/// must all be `Reg` before encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    VReg(VReg),
}

/// x86-64 machine instruction.
///
/// Operands are physical registers (`Operand::Reg`) after register allocation.
/// The encoder panics if it encounters a `VReg` operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MachInst {
    // ── Data movement ─────────────────────────────────────────────────────────
    MovRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    MovRI {
        size: OpSize,
        dst: Operand,
        imm: i64,
    },
    /// Load: dst = [addr]
    MovRM {
        size: OpSize,
        dst: Operand,
        addr: Addr,
    },
    /// Store: [addr] = src
    MovMR {
        size: OpSize,
        addr: Addr,
        src: Operand,
    },

    // ── ALU reg-reg ───────────────────────────────────────────────────────────
    AddRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    AddRI {
        size: OpSize,
        dst: Operand,
        imm: i32,
    },
    AddRM {
        size: OpSize,
        dst: Operand,
        addr: Addr,
    },
    SubRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    SubRI {
        size: OpSize,
        dst: Operand,
        imm: i32,
    },
    AndRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    OrRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    XorRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },

    // ── Shifts ────────────────────────────────────────────────────────────────
    ShlRI {
        size: OpSize,
        dst: Operand,
        imm: u8,
    },
    ShrRI {
        size: OpSize,
        dst: Operand,
        imm: u8,
    },
    SarRI {
        size: OpSize,
        dst: Operand,
        imm: u8,
    },
    /// Shift left by CL
    ShlRCL {
        size: OpSize,
        dst: Operand,
    },
    /// Shift right (logical) by CL
    ShrRCL {
        size: OpSize,
        dst: Operand,
    },
    /// Shift right (arithmetic) by CL
    SarRCL {
        size: OpSize,
        dst: Operand,
    },

    // ── Multiply ──────────────────────────────────────────────────────────────
    Imul2RR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    Imul3RRI {
        size: OpSize,
        dst: Operand,
        src: Operand,
        imm: i32,
    },

    // ── LEA ───────────────────────────────────────────────────────────────────
    Lea {
        size: OpSize,
        dst: Operand,
        addr: Addr,
    },

    // ── Compare / Test ────────────────────────────────────────────────────────
    CmpRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    CmpRI {
        size: OpSize,
        dst: Operand,
        imm: i32,
    },
    TestRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    TestRI {
        size: OpSize,
        dst: Operand,
        imm: i32,
    },

    // ── Stack ─────────────────────────────────────────────────────────────────
    Push {
        src: Operand,
    },
    Pop {
        dst: Operand,
    },

    // ── Control flow ──────────────────────────────────────────────────────────
    CallDirect {
        target: Symbol,
    },
    CallIndirect {
        target: Operand,
    },
    Ret,
    Jmp {
        target: LabelId,
    },
    Jcc {
        cc: CondCode,
        target: LabelId,
    },

    // ── Conditional ───────────────────────────────────────────────────────────
    Setcc {
        cc: CondCode,
        dst: Operand,
    },
    Cmov {
        size: OpSize,
        cc: CondCode,
        dst: Operand,
        src: Operand,
    },

    // ── Division support ──────────────────────────────────────────────────────
    /// Sign-extend EAX into EDX:EAX (32-bit)
    Cdq,
    /// Sign-extend RAX into RDX:RAX (64-bit)
    Cqo,
    /// CWD: sign-extend AX into DX:AX (16-bit)
    Cwd,
    /// CBW: sign-extend AL into AX (8-bit)
    Cbw,
    Idiv {
        size: OpSize,
        src: Operand,
    },
    Div {
        size: OpSize,
        src: Operand,
    },

    // ── Unary ─────────────────────────────────────────────────────────────────
    Neg {
        size: OpSize,
        dst: Operand,
    },
    Not {
        size: OpSize,
        dst: Operand,
    },
    Inc {
        size: OpSize,
        dst: Operand,
    },
    Dec {
        size: OpSize,
        dst: Operand,
    },

    // ── NOP ───────────────────────────────────────────────────────────────────
    Nop {
        size: u8,
    },

    // ── Zero/Sign extend ─────────────────────────────────────────────────────
    /// Zero-extend byte to 64-bit
    MovzxBR {
        dst: Operand,
        src: Operand,
    },
    /// Sign-extend byte to 64-bit
    MovsxBR {
        dst: Operand,
        src: Operand,
    },
    /// Zero-extend word to 64-bit
    MovzxWR {
        dst: Operand,
        src: Operand,
    },
    /// Zero-extend byte from memory to 64-bit register: MOVZX r64, byte ptr [addr]
    MovzxBRM {
        dst: Operand,
        addr: Addr,
    },
    /// Zero-extend word from memory to 64-bit register: MOVZX r64, word ptr [addr]
    MovzxWRM {
        dst: Operand,
        addr: Addr,
    },
    /// Sign-extend word to 64-bit
    MovsxWR {
        dst: Operand,
        src: Operand,
    },
    /// Sign-extend dword to qword (MOVSXD)
    MovsxDR {
        dst: Operand,
        src: Operand,
    },

    // ── SSE FP ────────────────────────────────────────────────────────────────
    MovsdRR {
        dst: Operand,
        src: Operand,
    },
    MovsdRM {
        dst: Operand,
        addr: Addr,
    },
    MovsdMR {
        addr: Addr,
        src: Operand,
    },
    MovssRR {
        dst: Operand,
        src: Operand,
    },
    MovssRM {
        dst: Operand,
        addr: Addr,
    },
    MovssMR {
        addr: Addr,
        src: Operand,
    },

    // ── FP arithmetic (double) ────────────────────────────────────────────────
    AddsdRR {
        dst: Operand,
        src: Operand,
    },
    SubsdRR {
        dst: Operand,
        src: Operand,
    },
    MulsdRR {
        dst: Operand,
        src: Operand,
    },
    DivsdRR {
        dst: Operand,
        src: Operand,
    },
    SqrtsdRR {
        dst: Operand,
        src: Operand,
    },

    // ── FP arithmetic (single) ────────────────────────────────────────────────
    AddssRR {
        dst: Operand,
        src: Operand,
    },
    SubssRR {
        dst: Operand,
        src: Operand,
    },
    MulssRR {
        dst: Operand,
        src: Operand,
    },
    DivssRR {
        dst: Operand,
        src: Operand,
    },
    SqrtssRR {
        dst: Operand,
        src: Operand,
    },

    // ── FP comparison ─────────────────────────────────────────────────────────
    UcomisdRR {
        src1: Operand,
        src2: Operand,
    },
    UcomissRR {
        src1: Operand,
        src2: Operand,
    },

    // ── SSE FP conversion ──────────────────────────────────────────────────────
    /// cvtsi2sd: GPR -> XMM (int -> f64)
    Cvtsi2sdRR {
        dst: Operand,
        src: Operand,
    },
    /// cvtsi2ss: GPR -> XMM (int -> f32)
    Cvtsi2ssRR {
        dst: Operand,
        src: Operand,
    },
    /// cvttsd2si: XMM -> GPR (f64 -> int, truncation)
    Cvttsd2siRR {
        dst: Operand,
        src: Operand,
    },
    /// cvttss2si: XMM -> GPR (f32 -> int, truncation)
    Cvttss2siRR {
        dst: Operand,
        src: Operand,
    },
    /// cvtsd2ss: XMM -> XMM (f64 -> f32)
    Cvtsd2ssRR {
        dst: Operand,
        src: Operand,
    },
    /// cvtss2sd: XMM -> XMM (f32 -> f64)
    Cvtss2sdRR {
        dst: Operand,
        src: Operand,
    },

    // ── Bitcast / MOVQ between GPR and XMM ───────────────────────────────────
    /// MOVQ xmm, r/m64  (66 REX.W 0F 6E /r) — move 64-bit integer into XMM.
    MovqToXmm {
        dst: Operand, // XMM
        src: Operand, // GPR
    },
    /// MOVQ r/m64, xmm  (66 REX.W 0F 7E /r) — move XMM bits into 64-bit integer register.
    MovqFromXmm {
        dst: Operand, // GPR
        src: Operand, // XMM
    },

    // ── RIP-relative LEA ─────────────────────────────────────────────────────
    /// LEA dst, [RIP + symbol] — load effective address of a global symbol.
    LeaRipRelative {
        dst: Operand,
        symbol: String,
    },
}
