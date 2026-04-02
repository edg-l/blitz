mod control;
mod gpr;
mod sse;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use crate::ir::condcode::CondCode;

use super::addr::Addr;
use super::inst::{LabelId, MachInst, OpSize, Operand};
use super::reg::Reg;

// ── Relocation ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelocKind {
    /// 32-bit PC-relative
    PC32,
    /// 32-bit PC-relative through PLT
    PLT32,
    /// 64-bit absolute
    Abs64,
    /// 32-bit absolute (signed)
    Abs32S,
}

#[derive(Debug, Clone)]
pub struct Reloc {
    /// Byte offset in the output buffer where the relocation applies.
    pub offset: usize,
    pub kind: RelocKind,
    pub symbol: String,
    pub addend: i64,
}

// ── Label fixups ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum FixupKind {
    Rel8,
    Rel32,
}

#[derive(Debug, Clone)]
struct Fixup {
    /// Byte offset of the rel8/rel32 field in the buffer.
    offset: usize,
    target: LabelId,
    kind: FixupKind,
}

// ── Encoder ───────────────────────────────────────────────────────────────────

pub struct Encoder {
    pub buf: Vec<u8>,
    labels: BTreeMap<LabelId, usize>,
    fixups: Vec<Fixup>,
    pub relocations: Vec<Reloc>,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            labels: BTreeMap::new(),
            fixups: Vec::new(),
            relocations: Vec::new(),
        }
    }

    // ── Low-level emit helpers ─────────────────────────────────────────────

    pub(super) fn emit_byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    pub(super) fn emit_le16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(super) fn emit_le32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(super) fn emit_le64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    // ── REX prefix helpers ────────────────────────────────────────────────

    /// Emit a REX prefix byte: 0100WRXB.
    pub(super) fn emit_rex(&mut self, w: bool, reg: u8, index: u8, base: u8) {
        let mut rex = 0x40u8;
        if w {
            rex |= 0x08;
        }
        if reg > 7 {
            rex |= 0x04; // REX.R
        }
        if index > 7 {
            rex |= 0x02; // REX.X
        }
        if base > 7 {
            rex |= 0x01; // REX.B
        }
        self.emit_byte(rex);
    }

    /// Only emit REX if any bit would be set.
    pub(super) fn maybe_emit_rex(&mut self, w: bool, reg: u8, index: u8, base: u8) {
        if w || reg > 7 || index > 7 || base > 7 {
            self.emit_rex(w, reg, index, base);
        }
    }

    // ── OpSize prefix/REX helpers ─────────────────────────────────────────

    /// Emit the 0x66 operand-size override prefix for S16. Must be called BEFORE REX.
    pub(super) fn emit_size_prefix(&mut self, size: OpSize) {
        if size == OpSize::S16 {
            self.emit_byte(0x66);
        }
    }

    /// Emit the appropriate REX prefix for the given operand size.
    ///
    /// - S64: REX.W (sets W bit)
    /// - S8: bare REX (0x40) if any operand has hw_enc 4-7 (SPL/BPL/SIL/DIL),
    ///   plus R/X/B extension bits as needed
    /// - S32/S16: only emit REX if R/X/B extension bits are needed
    pub(super) fn emit_rex_for_size(&mut self, size: OpSize, reg: u8, index: u8, base: u8) {
        match size {
            OpSize::S64 => {
                self.emit_rex(true, reg, index, base);
            }
            OpSize::S8 => {
                if Self::needs_byte_rex(reg)
                    || Self::needs_byte_rex(index)
                    || Self::needs_byte_rex(base)
                    || reg > 7
                    || index > 7
                    || base > 7
                {
                    self.emit_rex(false, reg, index, base);
                }
            }
            OpSize::S32 | OpSize::S16 => {
                self.maybe_emit_rex(false, reg, index, base);
            }
        }
    }

    /// Returns true if the register hw_enc 4-7 requires a REX prefix in byte
    /// operations to access SPL/BPL/SIL/DIL (instead of AH/CH/DH/BH).
    pub(super) fn needs_byte_rex(reg: u8) -> bool {
        (4..=7).contains(&reg)
    }

    // ── ModRM / SIB ───────────────────────────────────────────────────────

    pub(super) fn emit_modrm(&mut self, mod_: u8, reg: u8, rm: u8) {
        self.emit_byte((mod_ << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    pub(super) fn emit_sib(&mut self, scale: u8, index: u8, base: u8) {
        let ss = match scale {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => unreachable!("invalid SIB scale {scale}"),
        };
        self.emit_byte((ss << 6) | ((index & 7) << 3) | (base & 7));
    }

    /// Encode memory addressing after the opcode: emits ModRM (+ optional SIB
    /// + optional displacement).
    ///
    /// `reg_field` is the reg/opcode field in bits [5:3] of ModRM.
    pub(super) fn emit_addr(&mut self, reg_field: u8, addr: &Addr) {
        let disp = addr.disp;
        let has_disp = disp != 0;

        match (addr.base, addr.index) {
            // ── No base, no index: disp32 only (mod=00, rm=5 = RBP sentinel,
            //    SIB with base=5/index=4 means no-base + no-index).
            (None, None) => {
                // mod=00, rm=4 (SIB escape)
                self.emit_modrm(0b00, reg_field, 4);
                // SIB: scale=1, index=4 (none), base=5 (no base)
                self.emit_sib(1, 4, 5);
                self.emit_le32(disp);
            }

            // ── No base, with index: SIB with base=5 + disp32.
            (None, Some(idx)) => {
                let idx_enc = idx.hw_enc();
                // mod=00, rm=4 (SIB escape)
                self.emit_modrm(0b00, reg_field, 4);
                self.emit_sib(addr.scale, idx_enc, 5);
                self.emit_le32(disp);
            }

            // ── Base only (no index).
            (Some(base), None) => {
                let base_enc = base.hw_enc();

                if base_enc & 7 == 4 {
                    // RSP/R12: must use SIB even without an index.
                    let mod_ = if !has_disp {
                        0b00
                    } else if Self::fits_i8(disp) {
                        0b01
                    } else {
                        0b10
                    };
                    self.emit_modrm(mod_, reg_field, 4); // rm=4 = SIB escape
                    self.emit_sib(1, 4, base_enc); // index=4 = no index
                    match mod_ {
                        0b01 => self.emit_byte(disp as i8 as u8),
                        0b10 => self.emit_le32(disp),
                        _ => {}
                    }
                } else if base_enc & 7 == 5 {
                    // RBP/R13: mod=00 means RIP-relative; force mod=01 even with disp=0.
                    let mod_ = if has_disp && !Self::fits_i8(disp) {
                        0b10
                    } else {
                        0b01
                    };
                    self.emit_modrm(mod_, reg_field, base_enc);
                    match mod_ {
                        0b01 => self.emit_byte(disp as i8 as u8),
                        0b10 => self.emit_le32(disp),
                        _ => unreachable!(),
                    }
                } else {
                    // Normal base.
                    let mod_ = if !has_disp {
                        0b00
                    } else if Self::fits_i8(disp) {
                        0b01
                    } else {
                        0b10
                    };
                    self.emit_modrm(mod_, reg_field, base_enc);
                    match mod_ {
                        0b01 => self.emit_byte(disp as i8 as u8),
                        0b10 => self.emit_le32(disp),
                        _ => {}
                    }
                }
            }

            // ── Base + index.
            (Some(base), Some(idx)) => {
                let base_enc = base.hw_enc();
                let idx_enc = idx.hw_enc();

                let mod_ = if base_enc & 7 == 5 {
                    // RBP/R13 base with SIB: mod=00 still encodes disp32.
                    // But to encode zero displacement with RBP/R13 base we need mod=01.
                    if has_disp && !Self::fits_i8(disp) {
                        0b10
                    } else {
                        0b01
                    }
                } else if !has_disp {
                    0b00
                } else if Self::fits_i8(disp) {
                    0b01
                } else {
                    0b10
                };

                self.emit_modrm(mod_, reg_field, 4); // rm=4 = SIB escape
                self.emit_sib(addr.scale, idx_enc, base_enc);
                match mod_ {
                    0b01 => self.emit_byte(disp as i8 as u8),
                    0b10 => self.emit_le32(disp),
                    _ => {}
                }
            }
        }
    }

    // ── Immediate-size helpers ─────────────────────────────────────────────

    pub(super) fn fits_i8(v: i32) -> bool {
        (-128..=127).contains(&v)
    }

    pub(super) fn fits_i32(v: i64) -> bool {
        v >= i32::MIN as i64 && v <= i32::MAX as i64
    }

    // ── Operand extraction ────────────────────────────────────────────────

    pub(super) fn expect_reg(op: &Operand) -> Reg {
        match op {
            Operand::Reg(r) => *r,
            _ => panic!("expected physical register, got {op:?}"),
        }
    }

    // ── Condition code -> TTN byte ────────────────────────────────────────

    pub(super) fn cc_byte(cc: CondCode) -> u8 {
        match cc {
            CondCode::Eq => 0x4,  // JE/JZ
            CondCode::Ne => 0x5,  // JNE/JNZ
            CondCode::Ult => 0x2, // JB/JNAE/JC
            CondCode::Uge => 0x3, // JAE/JNB/JNC
            CondCode::Ule => 0x6, // JBE/JNA
            CondCode::Ugt => 0x7, // JA/JNBE
            CondCode::Slt => 0xC, // JL/JNGE
            CondCode::Sge => 0xD, // JGE/JNL
            CondCode::Sle => 0xE, // JLE/JNG
            CondCode::Sgt => 0xF, // JG/JNLE
        }
    }

    // ── Label management ──────────────────────────────────────────────────

    /// Bind a label to the current buffer position.
    pub fn bind_label(&mut self, label: LabelId) {
        self.labels.insert(label, self.buf.len());
    }

    /// Patch all recorded fixups using the bound label positions.
    pub fn resolve_fixups(&mut self) {
        for fixup in &self.fixups {
            let target = self.labels[&fixup.target];
            let offset = fixup.offset;
            match fixup.kind {
                FixupKind::Rel8 => {
                    let rel = (target as i64 - (offset as i64 + 1)) as i8;
                    self.buf[offset] = rel as u8;
                }
                FixupKind::Rel32 => {
                    let rel = (target as i64 - (offset as i64 + 4)) as i32;
                    self.buf[offset..offset + 4].copy_from_slice(&rel.to_le_bytes());
                }
            }
        }
    }

    // ── Dispatch with short/near form selection ────────────────────────────

    /// Encode `inst`, using the short (rel8) jump form when `short` is true
    /// for `Jmp`/`Jcc`.  All other instructions ignore `short`.
    pub fn encode_inst_with_form(&mut self, inst: &MachInst, short: bool) {
        match inst {
            MachInst::Jmp { target } => {
                if short {
                    self.encode_jmp_short(*target);
                } else {
                    self.encode_jmp(*target);
                }
            }
            MachInst::Jcc { cc, target } => {
                if short {
                    self.encode_jcc_short(*cc, *target);
                } else {
                    self.encode_jcc(*cc, *target);
                }
            }
            _ => self.encode_inst(inst),
        }
    }

    // ── Top-level dispatch ────────────────────────────────────────────────

    pub fn encode_inst(&mut self, inst: &MachInst) {
        match inst {
            MachInst::MovRR { size, dst, src } => {
                self.encode_mov_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovRI { size, dst, imm } => {
                self.encode_mov_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::MovRM { size, dst, addr } => {
                self.encode_mov_rm(*size, Self::expect_reg(dst), addr);
            }
            MachInst::MovMR { size, addr, src } => {
                self.encode_mov_mr(*size, addr, Self::expect_reg(src));
            }
            MachInst::AddRR { size, dst, src } => {
                self.encode_add_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::AddRI { size, dst, imm } => {
                self.encode_add_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::AddRM { size, dst, addr } => {
                self.encode_add_rm(*size, Self::expect_reg(dst), addr);
            }
            MachInst::SubRR { size, dst, src } => {
                self.encode_sub_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::SubRI { size, dst, imm } => {
                self.encode_sub_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::AndRR { size, dst, src } => {
                self.encode_and_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::OrRR { size, dst, src } => {
                self.encode_or_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::XorRR { size, dst, src } => {
                self.encode_xor_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::ShlRI { size, dst, imm } => {
                self.encode_shl_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::ShrRI { size, dst, imm } => {
                self.encode_shr_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::SarRI { size, dst, imm } => {
                self.encode_sar_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::ShlRCL { size, dst } => {
                self.encode_shl_rcl(*size, Self::expect_reg(dst));
            }
            MachInst::ShrRCL { size, dst } => {
                self.encode_shr_rcl(*size, Self::expect_reg(dst));
            }
            MachInst::SarRCL { size, dst } => {
                self.encode_sar_rcl(*size, Self::expect_reg(dst));
            }
            MachInst::Imul2RR { size, dst, src } => {
                self.encode_imul2_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::Imul3RRI {
                size,
                dst,
                src,
                imm,
            } => {
                self.encode_imul3_rri(*size, Self::expect_reg(dst), Self::expect_reg(src), *imm);
            }
            MachInst::Lea { size, dst, addr } => {
                self.encode_lea(*size, Self::expect_reg(dst), addr);
            }
            MachInst::CmpRR { size, dst, src } => {
                self.encode_cmp_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::CmpRI { size, dst, imm } => {
                self.encode_cmp_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::TestRR { size, dst, src } => {
                self.encode_test_rr(*size, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::TestRI { size, dst, imm } => {
                self.encode_test_ri(*size, Self::expect_reg(dst), *imm);
            }
            MachInst::Push { src } => {
                self.encode_push(Self::expect_reg(src));
            }
            MachInst::Pop { dst } => {
                self.encode_pop(Self::expect_reg(dst));
            }
            MachInst::CallDirect { target } => {
                self.encode_call_direct(target);
            }
            MachInst::CallIndirect { target } => {
                self.encode_call_indirect(Self::expect_reg(target));
            }
            MachInst::Ret => {
                self.encode_ret();
            }
            MachInst::Jmp { target } => {
                self.encode_jmp(*target);
            }
            MachInst::Jcc { cc, target } => {
                self.encode_jcc(*cc, *target);
            }
            MachInst::Setcc { cc, dst } => {
                self.encode_setcc(*cc, Self::expect_reg(dst));
            }
            MachInst::Cmov { size, cc, dst, src } => {
                self.encode_cmov(*size, *cc, Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::Cdq => {
                self.encode_cdq();
            }
            MachInst::Cqo => {
                self.encode_cqo();
            }
            MachInst::Cwd => {
                self.encode_cwd();
            }
            MachInst::Cbw => {
                self.encode_cbw();
            }
            MachInst::Idiv { size, src } => {
                self.encode_idiv(*size, Self::expect_reg(src));
            }
            MachInst::Div { size, src } => {
                self.encode_div(*size, Self::expect_reg(src));
            }
            MachInst::Neg { size, dst } => {
                self.encode_neg(*size, Self::expect_reg(dst));
            }
            MachInst::Not { size, dst } => {
                self.encode_not(*size, Self::expect_reg(dst));
            }
            MachInst::Inc { size, dst } => {
                self.encode_inc(*size, Self::expect_reg(dst));
            }
            MachInst::Dec { size, dst } => {
                self.encode_dec(*size, Self::expect_reg(dst));
            }
            MachInst::Nop { size } => {
                self.encode_nop(*size);
            }
            MachInst::MovzxBR { dst, src } => {
                self.encode_movzx_br(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovsxBR { dst, src } => {
                self.encode_movsx_br(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovzxWR { dst, src } => {
                self.encode_movzx_wr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovsxWR { dst, src } => {
                self.encode_movsx_wr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovzxBRM { dst, addr } => {
                self.encode_movzx_brm(Self::expect_reg(dst), addr);
            }
            MachInst::MovzxWRM { dst, addr } => {
                self.encode_movzx_wrm(Self::expect_reg(dst), addr);
            }
            MachInst::MovsxDR { dst, src } => {
                self.encode_movsx_dr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovsdRR { dst, src } => {
                self.encode_movsd_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovsdRM { dst, addr } => {
                self.encode_movsd_rm(Self::expect_reg(dst), addr);
            }
            MachInst::MovsdMR { addr, src } => {
                self.encode_movsd_mr(addr, Self::expect_reg(src));
            }
            MachInst::MovssRR { dst, src } => {
                self.encode_movss_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovssRM { dst, addr } => {
                self.encode_movss_rm(Self::expect_reg(dst), addr);
            }
            MachInst::MovssMR { addr, src } => {
                self.encode_movss_mr(addr, Self::expect_reg(src));
            }
            MachInst::AddsdRR { dst, src } => {
                self.encode_addsd_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::SubsdRR { dst, src } => {
                self.encode_subsd_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MulsdRR { dst, src } => {
                self.encode_mulsd_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::DivsdRR { dst, src } => {
                self.encode_divsd_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::SqrtsdRR { dst, src } => {
                self.encode_sqrtsd_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::AddssRR { dst, src } => {
                self.encode_addss_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::SubssRR { dst, src } => {
                self.encode_subss_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MulssRR { dst, src } => {
                self.encode_mulss_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::DivssRR { dst, src } => {
                self.encode_divss_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::SqrtssRR { dst, src } => {
                self.encode_sqrtss_rr(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::UcomisdRR { src1, src2 } => {
                self.encode_ucomisd_rr(Self::expect_reg(src1), Self::expect_reg(src2));
            }
            MachInst::UcomissRR { src1, src2 } => {
                self.encode_ucomiss_rr(Self::expect_reg(src1), Self::expect_reg(src2));
            }
            MachInst::MovqToXmm { dst, src } => {
                self.encode_movq_to_xmm(Self::expect_reg(dst), Self::expect_reg(src));
            }
            MachInst::MovqFromXmm { dst, src } => {
                self.encode_movq_from_xmm(Self::expect_reg(dst), Self::expect_reg(src));
            }
        }
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── inst_size ─────────────────────────────────────────────────────────────────

/// Return the encoded byte size of `inst` using a scratch encoder.
///
/// For `Jmp`/`Jcc` this returns the **near** (rel32) size; callers that need
/// the short size should use the constants in `emit::relax` directly.
pub fn inst_size(inst: &MachInst) -> usize {
    let mut scratch = Encoder::new();
    scratch.encode_inst(inst);
    scratch.buf.len()
}
