use crate::egraph::extract::VReg;
use crate::ir::condcode::CondCode;
use crate::ir::types::Type;

use super::abi::{CALLER_SAVED_GPR, CALLER_SAVED_XMM};
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
    pub fn from_int_type(ty: &Type) -> Self {
        match ty {
            Type::I8 => OpSize::S8,
            Type::I16 => OpSize::S16,
            Type::I32 => OpSize::S32,
            Type::I64 => OpSize::S64,
            _ => panic!("OpSize::from_int_type: not an integer type: {ty:?}"),
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
    /// Load: `dst = [addr]`
    MovRM {
        size: OpSize,
        dst: Operand,
        addr: Addr,
    },
    /// Store: `[addr] = src`
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
    AndRI {
        size: OpSize,
        dst: Operand,
        imm: i32,
    },
    OrRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    OrRI {
        size: OpSize,
        dst: Operand,
        imm: i32,
    },
    XorRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    XorRI {
        size: OpSize,
        dst: Operand,
        imm: i32,
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
    /// Rotate left by immediate count
    RolRI {
        size: OpSize,
        dst: Operand,
        imm: u8,
    },
    /// Double shift left by immediate count: `dst` shifts left by `imm` and the
    /// vacated low bits come from the high end of `src`, which is not written.
    ShldRRI {
        size: OpSize,
        dst: Operand,
        src: Operand,
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
    /// Zero-extend byte from memory to 64-bit register: `MOVZX r64, byte ptr [addr]`
    MovzxBRM {
        dst: Operand,
        addr: Addr,
    },
    /// Zero-extend word from memory to 64-bit register: `MOVZX r64, word ptr [addr]`
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
    /// cvtsi2sd: GPR -> XMM (int -> f64).
    /// `size` is the *source* width: `S32` for `cvtsi2sd xmm, r32`, `S64` for
    /// the REX.W form. Using S64 on a 32-bit value converts undefined high bits.
    Cvtsi2sdRR {
        size: OpSize,
        dst: Operand,
        src: Operand,
    },
    /// cvtsi2ss: GPR -> XMM (int -> f32). `size` is the source width.
    Cvtsi2ssRR {
        size: OpSize,
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

// ── Register effects ─────────────────────────────────────────────────────────

/// Physical registers an instruction reads and writes.
///
/// Used by the machine-level verifier (`crate::verify`) to check that nothing
/// reads a register before something writes it. Every bug it exists to catch
/// -- a folded addressing mode using the wrong index, a load reading a stale
/// register after its value was spilled, an address computation emitted after
/// the store that uses it -- shows up as exactly that.
///
/// Conventions:
/// - x86 two-operand ALU forms read *and* write `dst`, so it appears in both.
/// - `cmp`/`test`/`ucomis*` only write flags, so `dst` is a use and nothing is
///   defined.
/// - `mov`-like forms overwrite `dst`, so it is a def only.
/// - A memory operand contributes its base and index registers as uses.
/// - Calls are treated as defining every caller-saved register: after a call
///   those registers hold something, so reading one is not a use-before-def
///   even though the value is the callee's.
impl MachInst {
    /// The memory operand this instruction reads from, if any.
    ///
    /// `Lea` is excluded deliberately: it computes an address and reads no
    /// memory, so a `lea` naming a spill slot is not a reload of it.
    pub fn mem_load_addr(&self) -> Option<&Addr> {
        match self {
            MachInst::MovRM { addr, .. }
            | MachInst::AddRM { addr, .. }
            | MachInst::MovzxBRM { addr, .. }
            | MachInst::MovzxWRM { addr, .. }
            | MachInst::MovsdRM { addr, .. }
            | MachInst::MovssRM { addr, .. } => Some(addr),
            _ => None,
        }
    }

    /// The memory operand this instruction writes to, if any.
    pub fn mem_store_addr(&self) -> Option<&Addr> {
        match self {
            MachInst::MovMR { addr, .. }
            | MachInst::MovsdMR { addr, .. }
            | MachInst::MovssMR { addr, .. } => Some(addr),
            _ => None,
        }
    }

    /// Registers this instruction writes.
    pub fn defs(&self) -> Vec<Reg> {
        match self {
            MachInst::AddRR { dst, .. } => collect(&[dst]),
            MachInst::SubRR { dst, .. } => collect(&[dst]),
            MachInst::AndRR { dst, .. } => collect(&[dst]),
            MachInst::OrRR { dst, .. } => collect(&[dst]),
            MachInst::XorRR { dst, .. } => collect(&[dst]),
            MachInst::Imul2RR { dst, .. } => collect(&[dst]),
            MachInst::AddsdRR { dst, .. } => collect(&[dst]),
            MachInst::SubsdRR { dst, .. } => collect(&[dst]),
            MachInst::MulsdRR { dst, .. } => collect(&[dst]),
            MachInst::DivsdRR { dst, .. } => collect(&[dst]),
            MachInst::AddssRR { dst, .. } => collect(&[dst]),
            MachInst::SubssRR { dst, .. } => collect(&[dst]),
            MachInst::MulssRR { dst, .. } => collect(&[dst]),
            MachInst::DivssRR { dst, .. } => collect(&[dst]),
            MachInst::MovRR { dst, .. } => collect(&[dst]),
            MachInst::MovzxBR { dst, .. } => collect(&[dst]),
            MachInst::MovsxBR { dst, .. } => collect(&[dst]),
            MachInst::MovzxWR { dst, .. } => collect(&[dst]),
            MachInst::MovsxWR { dst, .. } => collect(&[dst]),
            MachInst::MovsxDR { dst, .. } => collect(&[dst]),
            MachInst::MovsdRR { dst, .. } => collect(&[dst]),
            MachInst::MovssRR { dst, .. } => collect(&[dst]),
            MachInst::SqrtsdRR { dst, .. } => collect(&[dst]),
            MachInst::SqrtssRR { dst, .. } => collect(&[dst]),
            MachInst::Cvttsd2siRR { dst, .. } => collect(&[dst]),
            MachInst::Cvttss2siRR { dst, .. } => collect(&[dst]),
            MachInst::Cvtsd2ssRR { dst, .. } => collect(&[dst]),
            MachInst::Cvtss2sdRR { dst, .. } => collect(&[dst]),
            MachInst::MovqToXmm { dst, .. } => collect(&[dst]),
            MachInst::MovqFromXmm { dst, .. } => collect(&[dst]),
            MachInst::Cvtsi2sdRR { dst, .. } => collect(&[dst]),
            MachInst::Cvtsi2ssRR { dst, .. } => collect(&[dst]),
            MachInst::Imul3RRI { dst, .. } => collect(&[dst]),
            MachInst::Cmov { dst, .. } => collect(&[dst]),
            MachInst::CmpRR { .. } => Vec::new(),
            MachInst::TestRR { .. } => Vec::new(),
            MachInst::CmpRI { .. } => Vec::new(),
            MachInst::TestRI { .. } => Vec::new(),
            MachInst::UcomisdRR { src1: _, src2: _ } => Vec::new(),
            MachInst::UcomissRR { src1: _, src2: _ } => Vec::new(),
            MachInst::AddRI { dst, .. } => collect(&[dst]),
            MachInst::SubRI { dst, .. } => collect(&[dst]),
            MachInst::AndRI { dst, .. } => collect(&[dst]),
            MachInst::OrRI { dst, .. } => collect(&[dst]),
            MachInst::XorRI { dst, .. } => collect(&[dst]),
            MachInst::ShlRI { dst, .. } => collect(&[dst]),
            MachInst::ShrRI { dst, .. } => collect(&[dst]),
            MachInst::SarRI { dst, .. } => collect(&[dst]),
            MachInst::RolRI { dst, .. } => collect(&[dst]),
            MachInst::ShldRRI { dst, .. } => collect(&[dst]),
            MachInst::Neg { dst, .. } => collect(&[dst]),
            MachInst::Not { dst, .. } => collect(&[dst]),
            MachInst::Inc { dst, .. } => collect(&[dst]),
            MachInst::Dec { dst, .. } => collect(&[dst]),
            MachInst::ShlRCL { dst, .. } => collect(&[dst]),
            MachInst::ShrRCL { dst, .. } => collect(&[dst]),
            MachInst::SarRCL { dst, .. } => collect(&[dst]),
            MachInst::MovRI { dst, .. } => collect(&[dst]),
            MachInst::Setcc { dst, .. } => collect(&[dst]),
            MachInst::MovRM { dst, .. } => collect(&[dst]),
            MachInst::Lea { dst, .. } => collect(&[dst]),
            MachInst::MovzxBRM { dst, .. } => collect(&[dst]),
            MachInst::MovzxWRM { dst, .. } => collect(&[dst]),
            MachInst::MovsdRM { dst, .. } => collect(&[dst]),
            MachInst::MovssRM { dst, .. } => collect(&[dst]),
            MachInst::AddRM { dst, .. } => collect(&[dst]),
            MachInst::MovMR { .. } => Vec::new(),
            MachInst::MovsdMR { .. } => Vec::new(),
            MachInst::MovssMR { .. } => Vec::new(),
            MachInst::Push { src: _ } => Vec::new(),
            MachInst::Pop { dst } => collect(&[dst]),
            MachInst::Idiv { .. } => vec![Reg::RAX, Reg::RDX],
            MachInst::Div { .. } => vec![Reg::RAX, Reg::RDX],
            MachInst::LeaRipRelative { dst, .. } => collect(&[dst]),
            MachInst::CallDirect { .. } | MachInst::CallIndirect { .. } => {
                let mut v: Vec<Reg> = CALLER_SAVED_GPR.to_vec();
                v.extend(CALLER_SAVED_XMM);
                v
            }
            // cdq/cqo sign-extend RAX into RDX; cbw/cwd widen within RAX.
            MachInst::Cdq | MachInst::Cqo | MachInst::Cwd => vec![Reg::RDX],
            MachInst::Cbw => vec![Reg::RAX],
            MachInst::Ret | MachInst::Jmp { .. } | MachInst::Jcc { .. } | MachInst::Nop { .. } => {
                Vec::new()
            }
        }
    }

    /// Registers this instruction reads.
    pub fn uses(&self) -> Vec<Reg> {
        // `xor r, r` and `sub r, r` are zeroing idioms: the result does not
        // depend on the previous contents, and every x86 implementation
        // recognizes them as such. Treating them as reads would report a
        // use-before-def on the standard way of materializing zero.
        if let MachInst::XorRR { dst, src, .. } | MachInst::SubRR { dst, src, .. } = self
            && dst == src
        {
            return Vec::new();
        }
        match self {
            MachInst::AddRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::SubRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::AndRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::OrRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::XorRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::Imul2RR { dst, src, .. } => collect(&[dst, src]),
            MachInst::AddsdRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::SubsdRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::MulsdRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::DivsdRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::AddssRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::SubssRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::MulssRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::DivssRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::MovRR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovzxBR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovsxBR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovzxWR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovsxWR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovsxDR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovsdRR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovssRR { dst: _, src, .. } => collect(&[src]),
            MachInst::SqrtsdRR { dst: _, src, .. } => collect(&[src]),
            MachInst::SqrtssRR { dst: _, src, .. } => collect(&[src]),
            MachInst::Cvttsd2siRR { dst: _, src, .. } => collect(&[src]),
            MachInst::Cvttss2siRR { dst: _, src, .. } => collect(&[src]),
            MachInst::Cvtsd2ssRR { dst: _, src, .. } => collect(&[src]),
            MachInst::Cvtss2sdRR { dst: _, src, .. } => collect(&[src]),
            MachInst::MovqToXmm { dst: _, src, .. } => collect(&[src]),
            MachInst::MovqFromXmm { dst: _, src, .. } => collect(&[src]),
            MachInst::Cvtsi2sdRR { dst: _, src, .. } => collect(&[src]),
            MachInst::Cvtsi2ssRR { dst: _, src, .. } => collect(&[src]),
            MachInst::Imul3RRI { dst: _, src, .. } => collect(&[src]),
            MachInst::Cmov { dst, src, .. } => collect(&[dst, src]),
            MachInst::CmpRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::TestRR { dst, src, .. } => collect(&[dst, src]),
            MachInst::CmpRI { dst, .. } => collect(&[dst]),
            MachInst::TestRI { dst, .. } => collect(&[dst]),
            MachInst::UcomisdRR { src1, src2 } => collect(&[src1, src2]),
            MachInst::UcomissRR { src1, src2 } => collect(&[src1, src2]),
            MachInst::AddRI { dst, .. } => collect(&[dst]),
            MachInst::SubRI { dst, .. } => collect(&[dst]),
            MachInst::AndRI { dst, .. } => collect(&[dst]),
            MachInst::OrRI { dst, .. } => collect(&[dst]),
            MachInst::XorRI { dst, .. } => collect(&[dst]),
            MachInst::ShlRI { dst, .. } => collect(&[dst]),
            MachInst::ShrRI { dst, .. } => collect(&[dst]),
            MachInst::SarRI { dst, .. } => collect(&[dst]),
            MachInst::RolRI { dst, .. } => collect(&[dst]),
            MachInst::ShldRRI { dst, src, .. } => collect(&[dst, src]),
            MachInst::Neg { dst, .. } => collect(&[dst]),
            MachInst::Not { dst, .. } => collect(&[dst]),
            MachInst::Inc { dst, .. } => collect(&[dst]),
            MachInst::Dec { dst, .. } => collect(&[dst]),
            MachInst::ShlRCL { dst, .. } => {
                let mut v = collect(&[dst]);
                v.push(Reg::RCX);
                v
            }
            MachInst::ShrRCL { dst, .. } => {
                let mut v = collect(&[dst]);
                v.push(Reg::RCX);
                v
            }
            MachInst::SarRCL { dst, .. } => {
                let mut v = collect(&[dst]);
                v.push(Reg::RCX);
                v
            }
            MachInst::MovRI { .. } => Vec::new(),
            MachInst::Setcc { .. } => Vec::new(),
            MachInst::MovRM { dst: _, addr, .. } => addr_regs(addr),
            MachInst::Lea { dst: _, addr, .. } => addr_regs(addr),
            MachInst::MovzxBRM { dst: _, addr, .. } => addr_regs(addr),
            MachInst::MovzxWRM { dst: _, addr, .. } => addr_regs(addr),
            MachInst::MovsdRM { dst: _, addr, .. } => addr_regs(addr),
            MachInst::MovssRM { dst: _, addr, .. } => addr_regs(addr),
            MachInst::AddRM { dst, addr, .. } => {
                let mut v = addr_regs(addr);
                v.extend(collect(&[dst]));
                v
            }
            MachInst::MovMR { addr, src, .. } => {
                let mut v = addr_regs(addr);
                v.extend(collect(&[src]));
                v
            }
            MachInst::MovsdMR { addr, src, .. } => {
                let mut v = addr_regs(addr);
                v.extend(collect(&[src]));
                v
            }
            MachInst::MovssMR { addr, src, .. } => {
                let mut v = addr_regs(addr);
                v.extend(collect(&[src]));
                v
            }
            MachInst::Push { src } => collect(&[src]),
            MachInst::Pop { dst: _ } => Vec::new(),
            MachInst::Idiv { src, .. } => {
                let mut v = collect(&[src]);
                v.extend([Reg::RAX, Reg::RDX]);
                v
            }
            MachInst::Div { src, .. } => {
                let mut v = collect(&[src]);
                v.extend([Reg::RAX, Reg::RDX]);
                v
            }
            MachInst::LeaRipRelative { .. } => Vec::new(),
            // Argument registers are not listed: how many a call reads depends
            // on the callee's signature, which is not on the instruction.
            MachInst::CallDirect { .. } => Vec::new(),
            MachInst::CallIndirect { target } => collect(&[target]),
            MachInst::Cdq | MachInst::Cqo | MachInst::Cwd | MachInst::Cbw => vec![Reg::RAX],
            MachInst::Ret | MachInst::Jmp { .. } | MachInst::Jcc { .. } | MachInst::Nop { .. } => {
                Vec::new()
            }
        }
    }
}

/// Physical registers in an operand list, skipping any that is still virtual.
fn collect(ops: &[&Operand]) -> Vec<Reg> {
    ops.iter()
        .filter_map(|o| match o {
            Operand::Reg(r) => Some(*r),
            Operand::VReg(_) => None,
        })
        .collect()
}

/// Base and index registers of a memory operand.
fn addr_regs(addr: &Addr) -> Vec<Reg> {
    addr.base.into_iter().chain(addr.index).collect()
}
