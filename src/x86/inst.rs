use crate::egraph::extract::VReg;
use crate::ir::condcode::CondCode;

use super::addr::Addr;
use super::reg::Reg;

pub type LabelId = u32;
pub type Symbol = String;

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
        dst: Operand,
        src: Operand,
    },
    MovRI {
        dst: Operand,
        imm: i64,
    },
    /// Load: dst = [addr]
    MovRM {
        dst: Operand,
        addr: Addr,
    },
    /// Store: [addr] = src
    MovMR {
        addr: Addr,
        src: Operand,
    },

    // ── ALU reg-reg ───────────────────────────────────────────────────────────
    AddRR {
        dst: Operand,
        src: Operand,
    },
    AddRI {
        dst: Operand,
        imm: i32,
    },
    AddRM {
        dst: Operand,
        addr: Addr,
    },
    SubRR {
        dst: Operand,
        src: Operand,
    },
    SubRI {
        dst: Operand,
        imm: i32,
    },
    AndRR {
        dst: Operand,
        src: Operand,
    },
    OrRR {
        dst: Operand,
        src: Operand,
    },
    XorRR {
        dst: Operand,
        src: Operand,
    },

    // ── Shifts ────────────────────────────────────────────────────────────────
    ShlRI {
        dst: Operand,
        imm: u8,
    },
    ShrRI {
        dst: Operand,
        imm: u8,
    },
    SarRI {
        dst: Operand,
        imm: u8,
    },
    /// Shift left by CL
    ShlRCL {
        dst: Operand,
    },
    /// Shift right (logical) by CL
    ShrRCL {
        dst: Operand,
    },
    /// Shift right (arithmetic) by CL
    SarRCL {
        dst: Operand,
    },

    // ── Multiply ──────────────────────────────────────────────────────────────
    Imul2RR {
        dst: Operand,
        src: Operand,
    },
    Imul3RRI {
        dst: Operand,
        src: Operand,
        imm: i32,
    },

    // ── LEA ───────────────────────────────────────────────────────────────────
    Lea {
        dst: Operand,
        addr: Addr,
    },

    // ── Compare / Test ────────────────────────────────────────────────────────
    CmpRR {
        dst: Operand,
        src: Operand,
    },
    CmpRI {
        dst: Operand,
        imm: i32,
    },
    TestRR {
        dst: Operand,
        src: Operand,
    },
    TestRI {
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
        cc: CondCode,
        dst: Operand,
        src: Operand,
    },

    // ── Division support ──────────────────────────────────────────────────────
    /// Sign-extend EAX into EDX:EAX (32-bit)
    Cdq,
    /// Sign-extend RAX into RDX:RAX (64-bit)
    Cqo,
    Idiv {
        src: Operand,
    },
    Div {
        src: Operand,
    },

    // ── Unary ─────────────────────────────────────────────────────────────────
    Neg {
        dst: Operand,
    },
    Not {
        dst: Operand,
    },
    Inc {
        dst: Operand,
    },
    Dec {
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
}
