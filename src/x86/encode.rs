use std::collections::HashMap;

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
    labels: HashMap<LabelId, usize>,
    fixups: Vec<Fixup>,
    pub relocations: Vec<Reloc>,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
            relocations: Vec::new(),
        }
    }

    // ── Low-level emit helpers ─────────────────────────────────────────────

    fn emit_byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    fn emit_le16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn emit_le32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn emit_le64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    // ── REX prefix helpers ────────────────────────────────────────────────

    /// Emit a REX prefix byte: 0100WRXB.
    fn emit_rex(&mut self, w: bool, reg: u8, index: u8, base: u8) {
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
    fn maybe_emit_rex(&mut self, w: bool, reg: u8, index: u8, base: u8) {
        if w || reg > 7 || index > 7 || base > 7 {
            self.emit_rex(w, reg, index, base);
        }
    }

    // ── OpSize prefix/REX helpers ─────────────────────────────────────────

    /// Emit the 0x66 operand-size override prefix for S16. Must be called BEFORE REX.
    fn emit_size_prefix(&mut self, size: OpSize) {
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
    fn emit_rex_for_size(&mut self, size: OpSize, reg: u8, index: u8, base: u8) {
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
    fn needs_byte_rex(reg: u8) -> bool {
        (4..=7).contains(&reg)
    }

    // ── ModRM / SIB ───────────────────────────────────────────────────────

    fn emit_modrm(&mut self, mod_: u8, reg: u8, rm: u8) {
        self.emit_byte((mod_ << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    fn emit_sib(&mut self, scale: u8, index: u8, base: u8) {
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
    fn emit_addr(&mut self, reg_field: u8, addr: &Addr) {
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

                if base_enc == 4 {
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
                } else if base_enc == 5 {
                    // RBP/R13: mod=00 means RIP-relative; force mod=01 even with disp=0.
                    let mod_ = if !has_disp {
                        0b01
                    } else if Self::fits_i8(disp) {
                        0b01
                    } else {
                        0b10
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

                let mod_ = if base_enc == 5 {
                    // RBP/R13 base with SIB: mod=00 still encodes disp32.
                    // But to encode zero displacement with RBP base we need mod=01.
                    if !has_disp {
                        0b01
                    } else if Self::fits_i8(disp) {
                        0b01
                    } else {
                        0b10
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

    fn fits_i8(v: i32) -> bool {
        v >= -128 && v <= 127
    }

    fn fits_i32(v: i64) -> bool {
        v >= i32::MIN as i64 && v <= i32::MAX as i64
    }

    // ── Operand extraction ────────────────────────────────────────────────

    fn expect_reg(op: &Operand) -> Reg {
        match op {
            Operand::Reg(r) => *r,
            _ => panic!("expected physical register, got {op:?}"),
        }
    }

    // ── Condition code -> TTN byte ────────────────────────────────────────

    fn cc_byte(cc: CondCode) -> u8 {
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

    // ── Instruction encoders ──────────────────────────────────────────────

    /// MOV r/m, r  — used as the canonical RR form.
    pub fn encode_mov_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, s, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, s, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, s, 0, d),
        }
        let opcode = if size == OpSize::S8 { 0x88 } else { 0x89 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, s, d);
    }

    /// MOV r, imm.
    pub fn encode_mov_ri(&mut self, size: OpSize, dst: Reg, imm: i64) {
        let d = dst.hw_enc();
        match size {
            OpSize::S64 => {
                if Self::fits_i32(imm) {
                    self.emit_rex(true, 0, 0, d);
                    self.emit_byte(0xC7);
                    self.emit_modrm(0b11, 0, d);
                    self.emit_le32(imm as i32);
                } else {
                    self.emit_rex(true, 0, 0, d);
                    self.emit_byte(0xB8 | (d & 7));
                    self.emit_le64(imm);
                }
            }
            OpSize::S32 => {
                // B8+rd + imm32 (no REX.W; upper 32 bits auto-zeroed)
                self.maybe_emit_rex(false, 0, 0, d);
                self.emit_byte(0xB8 | (d & 7));
                self.emit_le32(imm as i32);
            }
            OpSize::S16 => {
                self.emit_size_prefix(size);
                self.maybe_emit_rex(false, 0, 0, d);
                self.emit_byte(0xB8 | (d & 7));
                self.emit_le16(imm as u16);
            }
            OpSize::S8 => {
                self.emit_rex_for_size(size, 0, 0, d);
                self.emit_byte(0xB0 | (d & 7));
                self.emit_byte(imm as u8);
            }
        }
    }

    /// MOV r, r/m  (load from memory).
    pub fn encode_mov_rm(&mut self, size: OpSize, dst: Reg, addr: &Addr) {
        let d = dst.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, d, idx, base),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, d, idx, base),
            OpSize::S8 => self.emit_rex_for_size(size, d, idx, base),
        }
        let opcode = if size == OpSize::S8 { 0x8A } else { 0x8B };
        self.emit_byte(opcode);
        self.emit_addr(d, addr);
    }

    /// MOV r/m, r  (store to memory).
    pub fn encode_mov_mr(&mut self, size: OpSize, addr: &Addr, src: Reg) {
        let s = src.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, s, idx, base),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, s, idx, base),
            OpSize::S8 => self.emit_rex_for_size(size, s, idx, base),
        }
        let opcode = if size == OpSize::S8 { 0x88 } else { 0x89 };
        self.emit_byte(opcode);
        self.emit_addr(s, addr);
    }

    // ── ALU helpers ───────────────────────────────────────────────────────

    /// Encode a reg-reg ALU op.
    /// `opcode` is the r/m, r form (e.g. 0x01 for ADD).
    fn encode_alu_rr(&mut self, size: OpSize, opcode: u8, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, s, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, s, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, s, 0, d),
        }
        // S8 byte form: opcode & ~1 (e.g., ADD 0x01 -> 0x00)
        let op = if size == OpSize::S8 {
            opcode & !1
        } else {
            opcode
        };
        self.emit_byte(op);
        self.emit_modrm(0b11, s, d);
    }

    /// Encode reg-imm ALU op (ADD/SUB/AND/OR/XOR/CMP).
    /// `/n` is the ModRM /n value.
    /// `rax_shortcut` is the opcode for the RAX+imm32 shortcut.
    fn encode_alu_ri(&mut self, size: OpSize, slash_n: u8, rax_op: u8, dst: Reg, imm: i32) {
        let d = dst.hw_enc();

        if size == OpSize::S8 {
            // S8 ALWAYS uses 0x80 /n ib. Never 0x83.
            self.emit_rex_for_size(size, 0, 0, d);
            self.emit_byte(0x80);
            self.emit_modrm(0b11, slash_n, d);
            self.emit_byte(imm as u8);
            return;
        }

        // S16, S32, S64: emit operand-size prefix (0x66 for S16) and REX as needed.
        let needs_rex_w = size == OpSize::S64;
        self.emit_size_prefix(size);

        // Helper closure to emit the appropriate REX prefix.
        let emit_rex = |enc: &mut Self, reg_field: u8, rm_field: u8| {
            if needs_rex_w {
                enc.emit_rex(true, reg_field, 0, rm_field);
            } else {
                enc.maybe_emit_rex(false, reg_field, 0, rm_field);
            }
        };

        let imm_width = if size == OpSize::S16 { 2 } else { 4 };

        if Self::fits_i8(imm) {
            // Sign-extended imm8: 0x83 /n ib
            emit_rex(self, 0, d);
            self.emit_byte(0x83);
            self.emit_modrm(0b11, slash_n, d);
            self.emit_byte(imm as i8 as u8);
        } else if dst == Reg::RAX {
            // RAX short form: rax_op + imm16/imm32
            emit_rex(self, 0, 0);
            self.emit_byte(rax_op);
            if imm_width == 2 {
                self.emit_le16(imm as u16);
            } else {
                self.emit_le32(imm);
            }
        } else {
            // General form: 0x81 /n + imm16/imm32
            emit_rex(self, 0, d);
            self.emit_byte(0x81);
            self.emit_modrm(0b11, slash_n, d);
            if imm_width == 2 {
                self.emit_le16(imm as u16);
            } else {
                self.emit_le32(imm);
            }
        }
    }

    /// Encode reg-mem ALU op.
    /// `opcode` is the r, r/m form (e.g. 0x03 for ADD).
    fn encode_alu_rm(&mut self, size: OpSize, opcode: u8, dst: Reg, addr: &Addr) {
        let d = dst.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, d, idx, base),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, d, idx, base),
            OpSize::S8 => self.emit_rex_for_size(size, d, idx, base),
        }
        // S8: r,r/m form opcodes are also opcode & ~1 (e.g., ADD 0x03 -> 0x02)
        let op = if size == OpSize::S8 {
            opcode & !1
        } else {
            opcode
        };
        self.emit_byte(op);
        self.emit_addr(d, addr);
    }

    pub fn encode_add_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        self.encode_alu_rr(size, 0x01, dst, src);
    }

    pub fn encode_add_ri(&mut self, size: OpSize, dst: Reg, imm: i32) {
        self.encode_alu_ri(size, 0, 0x05, dst, imm);
    }

    pub fn encode_add_rm(&mut self, size: OpSize, dst: Reg, addr: &Addr) {
        self.encode_alu_rm(size, 0x03, dst, addr);
    }

    pub fn encode_sub_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        self.encode_alu_rr(size, 0x29, dst, src);
    }

    pub fn encode_sub_ri(&mut self, size: OpSize, dst: Reg, imm: i32) {
        self.encode_alu_ri(size, 5, 0x2D, dst, imm);
    }

    pub fn encode_and_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        self.encode_alu_rr(size, 0x21, dst, src);
    }

    pub fn encode_or_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        self.encode_alu_rr(size, 0x09, dst, src);
    }

    pub fn encode_xor_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        self.encode_alu_rr(size, 0x31, dst, src);
    }

    pub fn encode_cmp_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        self.encode_alu_rr(size, 0x39, dst, src);
    }

    pub fn encode_cmp_ri(&mut self, size: OpSize, dst: Reg, imm: i32) {
        self.encode_alu_ri(size, 7, 0x3D, dst, imm);
    }

    pub fn encode_test_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, s, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, s, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, s, 0, d),
        }
        let opcode = if size == OpSize::S8 { 0x84 } else { 0x85 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, s, d);
    }

    pub fn encode_test_ri(&mut self, size: OpSize, dst: Reg, imm: i32) {
        let d = dst.hw_enc();
        let w = size == OpSize::S64;
        match size {
            OpSize::S64 | OpSize::S32 | OpSize::S16 => {
                self.emit_size_prefix(size);
                if dst == Reg::RAX {
                    if w {
                        self.emit_rex(true, 0, 0, 0);
                    } else {
                        self.maybe_emit_rex(false, 0, 0, 0);
                    }
                    self.emit_byte(0xA9);
                    if size == OpSize::S16 {
                        self.emit_le16(imm as u16);
                    } else {
                        self.emit_le32(imm);
                    }
                } else {
                    if w {
                        self.emit_rex(true, 0, 0, d);
                    } else {
                        self.maybe_emit_rex(false, 0, 0, d);
                    }
                    self.emit_byte(0xF7);
                    self.emit_modrm(0b11, 0, d);
                    if size == OpSize::S16 {
                        self.emit_le16(imm as u16);
                    } else {
                        self.emit_le32(imm);
                    }
                }
            }
            OpSize::S8 => {
                // S8: 0xF6 /0 ib
                self.emit_rex_for_size(size, 0, 0, d);
                self.emit_byte(0xF6);
                self.emit_modrm(0b11, 0, d);
                self.emit_byte(imm as u8);
            }
        }
    }

    // ── Shifts ────────────────────────────────────────────────────────────

    fn encode_shift_ri(&mut self, size: OpSize, slash_n: u8, dst: Reg, imm: u8) {
        let d = dst.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, d),
        }
        if imm == 1 {
            // S8: 0xD0, S16/S32/S64: 0xD1
            let opcode = if size == OpSize::S8 { 0xD0 } else { 0xD1 };
            self.emit_byte(opcode);
            self.emit_modrm(0b11, slash_n, d);
        } else {
            // S8: 0xC0, S16/S32/S64: 0xC1
            let opcode = if size == OpSize::S8 { 0xC0 } else { 0xC1 };
            self.emit_byte(opcode);
            self.emit_modrm(0b11, slash_n, d);
            self.emit_byte(imm);
        }
    }

    fn encode_shift_cl(&mut self, size: OpSize, slash_n: u8, dst: Reg) {
        let d = dst.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, d),
        }
        // S8: 0xD2, S16/S32/S64: 0xD3
        let opcode = if size == OpSize::S8 { 0xD2 } else { 0xD3 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, slash_n, d);
    }

    pub fn encode_shl_ri(&mut self, size: OpSize, dst: Reg, imm: u8) {
        self.encode_shift_ri(size, 4, dst, imm);
    }

    pub fn encode_shr_ri(&mut self, size: OpSize, dst: Reg, imm: u8) {
        self.encode_shift_ri(size, 5, dst, imm);
    }

    pub fn encode_sar_ri(&mut self, size: OpSize, dst: Reg, imm: u8) {
        self.encode_shift_ri(size, 7, dst, imm);
    }

    pub fn encode_shl_rcl(&mut self, size: OpSize, dst: Reg) {
        self.encode_shift_cl(size, 4, dst);
    }

    pub fn encode_shr_rcl(&mut self, size: OpSize, dst: Reg) {
        self.encode_shift_cl(size, 5, dst);
    }

    pub fn encode_sar_rcl(&mut self, size: OpSize, dst: Reg) {
        self.encode_shift_cl(size, 7, dst);
    }

    // ── IMUL ──────────────────────────────────────────────────────────────

    pub fn encode_imul2_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        assert!(size != OpSize::S8, "IMUL has no byte form");
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, d, 0, s),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, d, 0, s),
            OpSize::S8 => unreachable!(),
        }
        self.emit_byte(0x0F);
        self.emit_byte(0xAF);
        self.emit_modrm(0b11, d, s);
    }

    pub fn encode_imul3_rri(&mut self, size: OpSize, dst: Reg, src: Reg, imm: i32) {
        assert!(size != OpSize::S8, "IMUL has no byte form");
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, d, 0, s),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, d, 0, s),
            OpSize::S8 => unreachable!(),
        }
        if Self::fits_i8(imm) {
            self.emit_byte(0x6B);
            self.emit_modrm(0b11, d, s);
            self.emit_byte(imm as i8 as u8);
        } else {
            self.emit_byte(0x69);
            self.emit_modrm(0b11, d, s);
            if size == OpSize::S16 {
                self.emit_le16(imm as u16);
            } else {
                self.emit_le32(imm);
            }
        }
    }

    // ── LEA ───────────────────────────────────────────────────────────────

    pub fn encode_lea(&mut self, size: OpSize, dst: Reg, addr: &Addr) {
        // LEA does not have a byte form; S8 makes no sense for LEA (addresses are >= 16-bit).
        // S16 is valid but rarely useful; we support it for completeness.
        assert!(size != OpSize::S8, "LEA has no byte form");
        let d = dst.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, d, idx, base),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, d, idx, base),
            OpSize::S8 => unreachable!(),
        }
        self.emit_byte(0x8D);
        self.emit_addr(d, addr);
    }

    // ── PUSH / POP ────────────────────────────────────────────────────────

    pub fn encode_push(&mut self, src: Reg) {
        let s = src.hw_enc();
        if s > 7 {
            // REX.B required for R8-R15
            self.emit_byte(0x41);
        }
        self.emit_byte(0x50 | (s & 7));
    }

    pub fn encode_pop(&mut self, dst: Reg) {
        let d = dst.hw_enc();
        if d > 7 {
            self.emit_byte(0x41);
        }
        self.emit_byte(0x58 | (d & 7));
    }

    // ── CALL ──────────────────────────────────────────────────────────────

    pub fn encode_call_direct(&mut self, target: &str) {
        // E8 + rel32 (filled with relocation)
        self.emit_byte(0xE8);
        let offset = self.buf.len();
        self.emit_le32(0); // placeholder
        self.relocations.push(Reloc {
            offset,
            kind: RelocKind::PLT32,
            symbol: target.to_string(),
            addend: -4,
        });
    }

    pub fn encode_call_indirect(&mut self, target: Reg) {
        // FF /2
        let t = target.hw_enc();
        self.maybe_emit_rex(false, 0, 0, t);
        self.emit_byte(0xFF);
        self.emit_modrm(0b11, 2, t);
    }

    // ── RET ───────────────────────────────────────────────────────────────

    pub fn encode_ret(&mut self) {
        self.emit_byte(0xC3);
    }

    // ── JMP ───────────────────────────────────────────────────────────────

    pub fn encode_jmp(&mut self, target: LabelId) {
        // Near form: E9 rel32
        self.emit_byte(0xE9);
        let offset = self.buf.len();
        self.emit_le32(0);
        self.fixups.push(Fixup {
            offset,
            target,
            kind: FixupKind::Rel32,
        });
    }

    /// Short (rel8) form: EB cb
    pub fn encode_jmp_short(&mut self, target: LabelId) {
        self.emit_byte(0xEB);
        let offset = self.buf.len();
        self.emit_byte(0); // placeholder rel8
        self.fixups.push(Fixup {
            offset,
            target,
            kind: FixupKind::Rel8,
        });
    }

    // ── Jcc ───────────────────────────────────────────────────────────────

    pub fn encode_jcc(&mut self, cc: CondCode, target: LabelId) {
        let tttn = Self::cc_byte(cc);
        // Near form: 0F 80+cc + rel32
        self.emit_byte(0x0F);
        self.emit_byte(0x80 | tttn);
        let offset = self.buf.len();
        self.emit_le32(0);
        self.fixups.push(Fixup {
            offset,
            target,
            kind: FixupKind::Rel32,
        });
    }

    /// Short (rel8) form: 7x cb
    pub fn encode_jcc_short(&mut self, cc: CondCode, target: LabelId) {
        let tttn = Self::cc_byte(cc);
        self.emit_byte(0x70 | tttn);
        let offset = self.buf.len();
        self.emit_byte(0); // placeholder rel8
        self.fixups.push(Fixup {
            offset,
            target,
            kind: FixupKind::Rel8,
        });
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

    // ── SETCC / CMOV ──────────────────────────────────────────────────────

    pub fn encode_setcc(&mut self, cc: CondCode, dst: Reg) {
        let tttn = Self::cc_byte(cc);
        let d = dst.hw_enc();
        if d > 7 {
            self.emit_byte(0x41); // REX.B for R8-R15
        } else if d >= 4 {
            self.emit_byte(0x40); // bare REX for SPL/BPL/SIL/DIL (hw_enc 4-7)
            // Without REX, hw_enc 4-7 address AH/CH/DH/BH.
        }
        self.emit_byte(0x0F);
        self.emit_byte(0x90 | tttn);
        self.emit_modrm(0b11, 0, d);
    }

    pub fn encode_cmov(&mut self, size: OpSize, cc: CondCode, dst: Reg, src: Reg) {
        assert!(size != OpSize::S8, "CMOV has no byte form");
        let tttn = Self::cc_byte(cc);
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, d, 0, s),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, d, 0, s),
            OpSize::S8 => unreachable!(),
        }
        self.emit_byte(0x0F);
        self.emit_byte(0x40 | tttn);
        self.emit_modrm(0b11, d, s);
    }

    // ── CDQ / CQO / IDIV / DIV / NEG / NOT ───────────────────────────────

    pub fn encode_cdq(&mut self) {
        // 99 (no REX.W — operates on 32-bit EAX/EDX)
        self.emit_byte(0x99);
    }

    pub fn encode_cqo(&mut self) {
        // REX.W + 99
        self.emit_byte(0x48);
        self.emit_byte(0x99);
    }

    /// CWD: sign-extend AX to DX:AX (0x66 0x99).
    pub fn encode_cwd(&mut self) {
        self.emit_byte(0x66);
        self.emit_byte(0x99);
    }

    /// CBW: sign-extend AL to AX (0x66 0x98).
    ///
    /// In 64-bit mode, bare opcode 0x98 is CWDE (sign-extend EAX to RAX).
    /// The 0x66 operand-size override prefix selects the 16-bit form (CBW),
    /// which sign-extends AL into AX as needed for 8-bit division setup.
    pub fn encode_cbw(&mut self) {
        self.emit_byte(0x66);
        self.emit_byte(0x98);
    }

    pub fn encode_idiv(&mut self, size: OpSize, src: Reg) {
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, s),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, s),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, s),
        }
        let opcode = if size == OpSize::S8 { 0xF6 } else { 0xF7 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, 7, s);
    }

    pub fn encode_div(&mut self, size: OpSize, src: Reg) {
        let s = src.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, s),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, s),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, s),
        }
        let opcode = if size == OpSize::S8 { 0xF6 } else { 0xF7 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, 6, s);
    }

    pub fn encode_neg(&mut self, size: OpSize, dst: Reg) {
        let d = dst.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, d),
        }
        let opcode = if size == OpSize::S8 { 0xF6 } else { 0xF7 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, 3, d);
    }

    pub fn encode_not(&mut self, size: OpSize, dst: Reg) {
        let d = dst.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, d),
        }
        let opcode = if size == OpSize::S8 { 0xF6 } else { 0xF7 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, 2, d);
    }

    /// INC r/m
    pub fn encode_inc(&mut self, size: OpSize, dst: Reg) {
        let d = dst.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, d),
        }
        let opcode = if size == OpSize::S8 { 0xFE } else { 0xFF };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, 0, d);
    }

    /// DEC r/m
    pub fn encode_dec(&mut self, size: OpSize, dst: Reg) {
        let d = dst.hw_enc();
        self.emit_size_prefix(size);
        match size {
            OpSize::S64 => self.emit_rex(true, 0, 0, d),
            OpSize::S32 | OpSize::S16 => self.maybe_emit_rex(false, 0, 0, d),
            OpSize::S8 => self.emit_rex_for_size(size, 0, 0, d),
        }
        let opcode = if size == OpSize::S8 { 0xFE } else { 0xFF };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, 1, d);
    }

    // ── MOVZX / MOVSX ─────────────────────────────────────────────────────

    pub fn encode_movzx_br(&mut self, dst: Reg, src: Reg) {
        // REX.W + 0F B6 /r (zero-extend byte to 64-bit)
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0xB6);
        self.emit_modrm(0b11, d, s);
    }

    pub fn encode_movsx_br(&mut self, dst: Reg, src: Reg) {
        // REX.W + 0F BE /r (sign-extend byte to 64-bit)
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0xBE);
        self.emit_modrm(0b11, d, s);
    }

    pub fn encode_movzx_wr(&mut self, dst: Reg, src: Reg) {
        // REX.W + 0F B7 /r (zero-extend word to 64-bit)
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0xB7);
        self.emit_modrm(0b11, d, s);
    }

    /// MOVZX r64, byte ptr [addr]: REX.W + 0F B6 /r with memory operand.
    pub fn encode_movzx_brm(&mut self, dst: Reg, addr: &Addr) {
        let d = dst.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_rex(true, d, idx, base);
        self.emit_byte(0x0F);
        self.emit_byte(0xB6);
        self.emit_addr(d, addr);
    }

    /// MOVZX r64, word ptr [addr]: REX.W + 0F B7 /r with memory operand.
    pub fn encode_movzx_wrm(&mut self, dst: Reg, addr: &Addr) {
        let d = dst.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_rex(true, d, idx, base);
        self.emit_byte(0x0F);
        self.emit_byte(0xB7);
        self.emit_addr(d, addr);
    }

    pub fn encode_movsx_wr(&mut self, dst: Reg, src: Reg) {
        // REX.W + 0F BF /r (sign-extend word to 64-bit)
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0xBF);
        self.emit_modrm(0b11, d, s);
    }

    pub fn encode_movsx_dr(&mut self, dst: Reg, src: Reg) {
        // REX.W + 63 /r (MOVSXD: sign-extend dword to qword)
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x63);
        self.emit_modrm(0b11, d, s);
    }

    // ── NOP ───────────────────────────────────────────────────────────────

    pub fn encode_nop(&mut self, size: u8) {
        // Intel-recommended multi-byte NOP sequences.
        match size {
            0 => {}
            1 => self.emit_byte(0x90),
            2 => {
                self.emit_byte(0x66);
                self.emit_byte(0x90);
            }
            3 => {
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x00);
            }
            4 => {
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x40);
                self.emit_byte(0x00);
            }
            5 => {
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x44);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            6 => {
                self.emit_byte(0x66);
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x44);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            7 => {
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x80);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            8 => {
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x84);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            9 => {
                self.emit_byte(0x66);
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x84);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            _ => {
                // For larger sizes, repeat 9-byte NOPs then fill the rest.
                let mut remaining = size;
                while remaining >= 9 {
                    self.encode_nop(9);
                    remaining -= 9;
                }
                if remaining > 0 {
                    self.encode_nop(remaining);
                }
            }
        }
    }

    // ── SSE FP helpers ────────────────────────────────────────────────────

    /// Emit a 2-byte SSE mandatory prefix + 0F opcode + ModRM for reg-reg.
    /// `prefix` is 0xF2 (SD) or 0xF3 (SS), or 0x66 (packed/UCOMISD).
    /// `opcode` is the second byte after 0x0F.
    fn encode_sse_rr(&mut self, prefix: u8, opcode: u8, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_byte(prefix);
        self.maybe_emit_rex(false, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_modrm(0b11, d, s);
    }

    /// Emit SSE load: prefix + REX + 0F opcode + addr.
    fn encode_sse_rm(&mut self, prefix: u8, opcode: u8, dst: Reg, addr: &Addr) {
        let d = dst.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_byte(prefix);
        self.maybe_emit_rex(false, d, idx, base);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_addr(d, addr);
    }

    /// Emit SSE store: prefix + REX + 0F opcode + addr (src in reg field).
    fn encode_sse_mr(&mut self, prefix: u8, opcode: u8, addr: &Addr, src: Reg) {
        let s = src.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_byte(prefix);
        self.maybe_emit_rex(false, s, idx, base);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_addr(s, addr);
    }

    pub fn encode_movsd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF2, 0x10, dst, src);
    }

    pub fn encode_movsd_rm(&mut self, dst: Reg, addr: &Addr) {
        self.encode_sse_rm(0xF2, 0x10, dst, addr);
    }

    pub fn encode_movsd_mr(&mut self, addr: &Addr, src: Reg) {
        self.encode_sse_mr(0xF2, 0x11, addr, src);
    }

    pub fn encode_movss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF3, 0x10, dst, src);
    }

    pub fn encode_movss_rm(&mut self, dst: Reg, addr: &Addr) {
        self.encode_sse_rm(0xF3, 0x10, dst, addr);
    }

    pub fn encode_movss_mr(&mut self, addr: &Addr, src: Reg) {
        self.encode_sse_mr(0xF3, 0x11, addr, src);
    }

    pub fn encode_addsd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF2, 0x58, dst, src);
    }

    pub fn encode_subsd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF2, 0x5C, dst, src);
    }

    pub fn encode_mulsd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF2, 0x59, dst, src);
    }

    pub fn encode_divsd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF2, 0x5E, dst, src);
    }

    pub fn encode_sqrtsd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF2, 0x51, dst, src);
    }

    pub fn encode_addss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF3, 0x58, dst, src);
    }

    pub fn encode_subss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF3, 0x5C, dst, src);
    }

    pub fn encode_mulss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF3, 0x59, dst, src);
    }

    pub fn encode_divss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF3, 0x5E, dst, src);
    }

    pub fn encode_sqrtss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF3, 0x51, dst, src);
    }

    pub fn encode_ucomisd_rr(&mut self, src1: Reg, src2: Reg) {
        // 66 0F 2E /r
        self.encode_sse_rr(0x66, 0x2E, src1, src2);
    }

    /// MOVQ xmm, r/m64 — move 64-bit integer from GPR into the low 64 bits of XMM.
    /// Encoding: 66 REX.W 0F 6E /r  (reg field = xmm, rm = gpr)
    pub fn encode_movq_to_xmm(&mut self, dst: Reg, src: Reg) {
        let d = dst.hw_enc(); // XMM register
        let s = src.hw_enc(); // GPR
        self.emit_byte(0x66);
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0x6E);
        self.emit_modrm(0b11, d, s);
    }

    /// MOVQ r/m64, xmm — move 64-bit value from XMM into GPR.
    /// Encoding: 66 REX.W 0F 7E /r  (reg field = xmm, rm = gpr)
    pub fn encode_movq_from_xmm(&mut self, dst: Reg, src: Reg) {
        let d = dst.hw_enc(); // GPR
        let s = src.hw_enc(); // XMM register
        self.emit_byte(0x66);
        self.emit_rex(true, s, 0, d);
        self.emit_byte(0x0F);
        self.emit_byte(0x7E);
        self.emit_modrm(0b11, s, d);
    }

    pub fn encode_ucomiss_rr(&mut self, src1: Reg, src2: Reg) {
        // 0F 2E /r (no mandatory prefix)
        let d = src1.hw_enc();
        let s = src2.hw_enc();
        self.maybe_emit_rex(false, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0x2E);
        self.emit_modrm(0b11, d, s);
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::condcode::CondCode;

    fn enc() -> Encoder {
        Encoder::new()
    }

    // Helper: assert exact byte sequence.
    fn check(bytes: &[u8], expected: &[u8]) {
        assert_eq!(
            bytes, expected,
            "bytes: {:02X?}\nexpected: {:02X?}",
            bytes, expected
        );
    }

    // 7.7 MOV ────────────────────────────────────────────────────────────────

    #[test]
    fn mov_rr_rax_rcx() {
        // mov rax, rcx  →  REX.W(48) + 89 /r  ModRM(C8) = mod11 reg1(rcx) rm0(rax)
        let mut e = enc();
        e.encode_mov_rr(OpSize::S64, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x48, 0x89, 0xC8]);
    }

    #[test]
    fn mov_rr_r8_r15() {
        // mov r8, r15  →  REX.W+R+B(4D) + 89 + ModRM
        let mut e = enc();
        e.encode_mov_rr(OpSize::S64, Reg::R8, Reg::R15);
        // REX: W=1, R=1(r15 hw_enc=15>7), B=1(r8 hw_enc=8>7) => 0x4D
        // opcode 0x89, ModRM mod=11, reg=7(r15&7), rm=0(r8&7) => 0b11_111_000 = 0xF8
        check(&e.buf, &[0x4D, 0x89, 0xF8]);
    }

    #[test]
    fn mov_ri_small_imm() {
        // mov rax, 42  →  REX.W(48) + C7 /0 + imm32(2A 00 00 00)
        let mut e = enc();
        e.encode_mov_ri(OpSize::S64, Reg::RAX, 42);
        check(&e.buf, &[0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn mov_ri_large_imm() {
        // mov rax, 0x100000000  →  REX.W(48) + B8 + imm64
        let mut e = enc();
        e.encode_mov_ri(OpSize::S64, Reg::RAX, 0x1_0000_0000i64);
        assert_eq!(e.buf[0], 0x48);
        assert_eq!(e.buf[1], 0xB8);
        assert_eq!(&e.buf[2..], &0x1_0000_0000i64.to_le_bytes());
    }

    // 7.8 ADD ────────────────────────────────────────────────────────────────

    #[test]
    fn add_rr_rax_rcx() {
        // add rax, rcx  →  REX.W(48) + 01 + ModRM(C8)
        // opcode 01: ADD r/m64, r64; reg=rcx(1), rm=rax(0)
        let mut e = enc();
        e.encode_add_rr(OpSize::S64, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x48, 0x01, 0xC8]);
    }

    #[test]
    fn add_rr_r12_r15() {
        // add r12, r15
        // REX: W=1, R=1(r15), B=1(r12) => 0x4D
        // ModRM: mod=11, reg=7(r15&7), rm=4(r12&7) => 0b11_111_100 = 0xFC
        let mut e = enc();
        e.encode_add_rr(OpSize::S64, Reg::R12, Reg::R15);
        check(&e.buf, &[0x4D, 0x01, 0xFC]);
    }

    #[test]
    fn add_ri_small() {
        // add rax, 5  →  REX.W(48) + 83 /0 + 05
        let mut e = enc();
        e.encode_add_ri(OpSize::S64, Reg::RAX, 5);
        check(&e.buf, &[0x48, 0x83, 0xC0, 0x05]);
    }

    #[test]
    fn add_ri_large() {
        // add rax, 100000  →  REX.W(48) + 05 + imm32 (RAX shortcut)
        let mut e = enc();
        e.encode_add_ri(OpSize::S64, Reg::RAX, 100000);
        let mut expected = vec![0x48, 0x05];
        expected.extend_from_slice(&100000i32.to_le_bytes());
        check(&e.buf, &expected);
    }

    #[test]
    fn add_ri_large_non_rax() {
        // add rcx, 100000  →  REX.W(48) + 81 /0 + imm32
        let mut e = enc();
        e.encode_add_ri(OpSize::S64, Reg::RCX, 100000);
        let mut expected = vec![0x48, 0x81, 0xC1]; // ModRM: mod=11, /0=0, rm=1(rcx)
        expected.extend_from_slice(&100000i32.to_le_bytes());
        check(&e.buf, &expected);
    }

    #[test]
    fn sub_rr() {
        // sub rax, rcx  →  REX.W(48) + 29 + ModRM(C8)
        let mut e = enc();
        e.encode_sub_rr(OpSize::S64, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x48, 0x29, 0xC8]);
    }

    #[test]
    fn xor_rr() {
        // xor rax, rax  →  REX.W(48) + 31 + ModRM(C0)
        let mut e = enc();
        e.encode_xor_rr(OpSize::S64, Reg::RAX, Reg::RAX);
        check(&e.buf, &[0x48, 0x31, 0xC0]);
    }

    // 7.9 Shifts ─────────────────────────────────────────────────────────────

    #[test]
    fn shl_ri() {
        // shl rax, 3  →  REX.W(48) + C1 /4 + 03
        let mut e = enc();
        e.encode_shl_ri(OpSize::S64, Reg::RAX, 3);
        check(&e.buf, &[0x48, 0xC1, 0xE0, 0x03]);
    }

    #[test]
    fn shr_ri_by_1() {
        // shr rax, 1  →  REX.W(48) + D1 /5
        let mut e = enc();
        e.encode_shr_ri(OpSize::S64, Reg::RAX, 1);
        check(&e.buf, &[0x48, 0xD1, 0xE8]);
    }

    #[test]
    fn sar_rcl() {
        // sar rcx, cl  →  REX.W(48) + D3 /7 + ModRM(mod11,/7,rm1)
        let mut e = enc();
        e.encode_sar_rcl(OpSize::S64, Reg::RCX);
        // ModRM: mod=11, /7=7, rm=1(rcx) => 0b11_111_001 = 0xF9
        check(&e.buf, &[0x48, 0xD3, 0xF9]);
    }

    // 7.10 LEA ───────────────────────────────────────────────────────────────

    #[test]
    fn lea_base_disp() {
        // lea rax, [rbx+4]  →  REX.W(48) + 8D + ModRM(mod01,reg0(rax),rm3(rbx)) + disp8(04)
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RBX), None, 1, 4);
        e.encode_lea(OpSize::S64, Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x8D, 0x43, 0x04]);
    }

    // 7.11 TEST ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rr() {
        // test rax, rcx  →  REX.W(48) + 85 + ModRM(C8)
        let mut e = enc();
        e.encode_test_rr(OpSize::S64, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x48, 0x85, 0xC8]);
    }

    #[test]
    fn test_ri_rax() {
        // test rax, 0xFF  →  REX.W(48) + A9 + imm32
        let mut e = enc();
        e.encode_test_ri(OpSize::S64, Reg::RAX, 0xFF);
        let mut expected = vec![0x48, 0xA9];
        expected.extend_from_slice(&0xFFi32.to_le_bytes());
        check(&e.buf, &expected);
    }

    // 7.12 IMUL ──────────────────────────────────────────────────────────────

    #[test]
    fn imul2_rr() {
        // imul rax, rcx  →  REX.W(48) + 0F AF + ModRM(mod11,reg0,rm1)
        let mut e = enc();
        e.encode_imul2_rr(OpSize::S64, Reg::RAX, Reg::RCX);
        // ModRM: mod=11, reg=0(rax), rm=1(rcx) => 0xC1
        check(&e.buf, &[0x48, 0x0F, 0xAF, 0xC1]);
    }

    #[test]
    fn imul3_rri_small() {
        // imul rax, rcx, 5  →  REX.W(48) + 6B + ModRM(C1) + 05
        let mut e = enc();
        e.encode_imul3_rri(OpSize::S64, Reg::RAX, Reg::RCX, 5);
        check(&e.buf, &[0x48, 0x6B, 0xC1, 0x05]);
    }

    // 7.13 PUSH/POP ──────────────────────────────────────────────────────────

    #[test]
    fn push_rax() {
        let mut e = enc();
        e.encode_push(Reg::RAX);
        check(&e.buf, &[0x50]);
    }

    #[test]
    fn push_r8() {
        // push r8  →  REX.B(41) + 50
        let mut e = enc();
        e.encode_push(Reg::R8);
        check(&e.buf, &[0x41, 0x50]);
    }

    #[test]
    fn pop_rbx() {
        let mut e = enc();
        e.encode_pop(Reg::RBX);
        check(&e.buf, &[0x5B]);
    }

    #[test]
    fn pop_r15() {
        // pop r15  →  REX.B(41) + 5F
        let mut e = enc();
        e.encode_pop(Reg::R15);
        check(&e.buf, &[0x41, 0x5F]);
    }

    // 7.14 CALL ──────────────────────────────────────────────────────────────

    #[test]
    fn call_indirect() {
        // call rax  →  FF /2  ModRM(mod11, /2, rm0) = 0xD0
        let mut e = enc();
        e.encode_call_indirect(Reg::RAX);
        check(&e.buf, &[0xFF, 0xD0]);
    }

    #[test]
    fn call_indirect_r11() {
        // call r11  →  REX.B(41) + FF /2  ModRM(mod11, /2, rm3) = 0xD3
        let mut e = enc();
        e.encode_call_indirect(Reg::R11);
        check(&e.buf, &[0x41, 0xFF, 0xD3]);
    }

    // 7.15 RET ───────────────────────────────────────────────────────────────

    #[test]
    fn ret() {
        let mut e = enc();
        e.encode_ret();
        check(&e.buf, &[0xC3]);
    }

    // 7.16 JMP (label fixup) ─────────────────────────────────────────────────

    #[test]
    fn jmp_label_forward() {
        let mut e = enc();
        e.encode_jmp(0);
        // resolve: target is right after the instruction (rel32=0)
        e.bind_label(0);
        e.resolve_fixups();
        // E9 + rel32(0) => forward jump past itself
        check(&e.buf, &[0xE9, 0x00, 0x00, 0x00, 0x00]);
    }

    // 7.17 Jcc ───────────────────────────────────────────────────────────────

    #[test]
    fn jcc_eq_label() {
        let mut e = enc();
        e.encode_jcc(CondCode::Eq, 1);
        e.bind_label(1);
        e.resolve_fixups();
        // 0F 84 + rel32(0)
        check(&e.buf, &[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    }

    // 7.18 SETCC / CMOV ──────────────────────────────────────────────────────

    #[test]
    fn setcc_eq_rax() {
        // sete al  →  0F 94 + ModRM(mod11, 0, rm0)
        let mut e = enc();
        e.encode_setcc(CondCode::Eq, Reg::RAX);
        check(&e.buf, &[0x0F, 0x94, 0xC0]);
    }

    #[test]
    fn cmov_eq() {
        // cmove rax, rcx  →  REX.W(48) + 0F 44 + ModRM(mod11,reg0(rax),rm1(rcx))
        let mut e = enc();
        e.encode_cmov(OpSize::S64, CondCode::Eq, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x48, 0x0F, 0x44, 0xC1]);
    }

    // 7.19 CDQ/CQO/IDIV/DIV/NEG/NOT ──────────────────────────────────────────

    #[test]
    fn cdq() {
        let mut e = enc();
        e.encode_cdq();
        check(&e.buf, &[0x99]);
    }

    #[test]
    fn cqo() {
        let mut e = enc();
        e.encode_cqo();
        check(&e.buf, &[0x48, 0x99]);
    }

    #[test]
    fn idiv_rcx() {
        // idiv rcx  →  REX.W(48) + F7 /7  ModRM(mod11, /7=7, rm1(rcx))
        // ModRM = 0b11_111_001 = 0xF9
        let mut e = enc();
        e.encode_idiv(OpSize::S64, Reg::RCX);
        check(&e.buf, &[0x48, 0xF7, 0xF9]);
    }

    #[test]
    fn neg_rax() {
        // neg rax  →  REX.W(48) + F7 /3  ModRM(mod11, /3=3, rm0(rax)) = 0xD8
        let mut e = enc();
        e.encode_neg(OpSize::S64, Reg::RAX);
        check(&e.buf, &[0x48, 0xF7, 0xD8]);
    }

    #[test]
    fn not_rax() {
        // not rax  →  REX.W(48) + F7 /2  ModRM(mod11, /2=2, rm0(rax)) = 0xD0
        let mut e = enc();
        e.encode_not(OpSize::S64, Reg::RAX);
        check(&e.buf, &[0x48, 0xF7, 0xD0]);
    }

    // 7.20 MOVZX/MOVSX ───────────────────────────────────────────────────────

    #[test]
    fn movzx_br() {
        // movzx rax, cl  →  REX.W(48) + 0F B6 + ModRM(mod11,reg0,rm1)
        let mut e = enc();
        e.encode_movzx_br(Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x48, 0x0F, 0xB6, 0xC1]);
    }

    #[test]
    fn movsx_dr() {
        // movsxd rax, ecx  →  REX.W(48) + 63 + ModRM(mod11,reg0,rm1)
        let mut e = enc();
        e.encode_movsx_dr(Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x48, 0x63, 0xC1]);
    }

    #[test]
    fn movzx_brm_base_disp() {
        // movzx rax, byte ptr [rcx + 8]
        // REX.W(48) + 0F B6 + ModRM(mod01,reg0,rm1) + disp8(08)
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RCX), None, 1, 8);
        e.encode_movzx_brm(Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x0F, 0xB6, 0x41, 0x08]);
    }

    #[test]
    fn movzx_brm_base_index_scale() {
        // movzx rax, byte ptr [rbx + rcx*4 + 16]
        // REX.W(48) + 0F B6 + ModRM(mod01,reg0,rm4=SIB) + SIB(ss2,idx1,base3) + disp8(10)
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RBX), Some(Reg::RCX), 4, 16);
        e.encode_movzx_brm(Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x0F, 0xB6, 0x44, 0x8B, 0x10]);
    }

    #[test]
    fn movzx_wrm_base_disp() {
        // movzx rax, word ptr [rcx + 8]
        // REX.W(48) + 0F B7 + ModRM(mod01,reg0,rm1) + disp8(08)
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RCX), None, 1, 8);
        e.encode_movzx_wrm(Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x0F, 0xB7, 0x41, 0x08]);
    }

    #[test]
    fn movzx_wrm_base_index_scale() {
        // movzx rax, word ptr [rbx + rcx*4 + 16]
        // REX.W(48) + 0F B7 + ModRM(mod01,reg0,rm4=SIB) + SIB(ss2,idx1,base3) + disp8(10)
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RBX), Some(Reg::RCX), 4, 16);
        e.encode_movzx_wrm(Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x0F, 0xB7, 0x44, 0x8B, 0x10]);
    }

    #[test]
    fn movzx_brm_r8_base_disp() {
        // movzx rax, byte ptr [r8 + 4]
        // REX.WB(49) + 0F B6 + ModRM(mod01,reg0,rm0=r8&7) + disp8(04)
        let mut e = enc();
        let addr = Addr::new(Some(Reg::R8), None, 1, 4);
        e.encode_movzx_brm(Reg::RAX, &addr);
        check(&e.buf, &[0x49, 0x0F, 0xB6, 0x40, 0x04]);
    }

    // 7.21 NOP ───────────────────────────────────────────────────────────────

    #[test]
    fn nop_1() {
        let mut e = enc();
        e.encode_nop(1);
        check(&e.buf, &[0x90]);
    }

    #[test]
    fn nop_3() {
        let mut e = enc();
        e.encode_nop(3);
        check(&e.buf, &[0x0F, 0x1F, 0x00]);
    }

    // 7.22 SSE FP ────────────────────────────────────────────────────────────

    #[test]
    fn movsd_rr() {
        // movsd xmm0, xmm1  →  F2 0F 10 ModRM(mod11,reg0,rm1)
        let mut e = enc();
        e.encode_movsd_rr(Reg::XMM0, Reg::XMM1);
        check(&e.buf, &[0xF2, 0x0F, 0x10, 0xC1]);
    }

    #[test]
    fn movsd_rr_xmm8_xmm15() {
        // movsd xmm8, xmm15  →  F2 REX.RB(45) 0F 10 ModRM(mod11,reg0,rm7)
        let mut e = enc();
        e.encode_movsd_rr(Reg::XMM8, Reg::XMM15);
        // REX: R=1(xmm8 hw=8), B=1(xmm15 hw=15) => 0x45
        check(&e.buf, &[0xF2, 0x45, 0x0F, 0x10, 0xC7]);
    }

    #[test]
    fn addsd_rr() {
        // addsd xmm0, xmm1  →  F2 0F 58 ModRM(C1)
        let mut e = enc();
        e.encode_addsd_rr(Reg::XMM0, Reg::XMM1);
        check(&e.buf, &[0xF2, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn ucomisd_rr() {
        // ucomisd xmm0, xmm1  →  66 0F 2E ModRM(C1)
        let mut e = enc();
        e.encode_ucomisd_rr(Reg::XMM0, Reg::XMM1);
        check(&e.buf, &[0x66, 0x0F, 0x2E, 0xC1]);
    }

    #[test]
    fn ucomiss_rr() {
        // ucomiss xmm0, xmm1  →  0F 2E ModRM(C1)
        let mut e = enc();
        e.encode_ucomiss_rr(Reg::XMM0, Reg::XMM1);
        check(&e.buf, &[0x0F, 0x2E, 0xC1]);
    }

    // Addressing mode special cases ──────────────────────────────────────────

    #[test]
    fn mov_rm_rsp_base() {
        // mov rax, [rsp+8]  — RSP base requires SIB
        // REX.W(48) + 8B + ModRM(mod01,reg0,rm4=SIB) + SIB(ss0,idx4,base4) + disp8(08)
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RSP), None, 1, 8);
        e.encode_mov_rm(OpSize::S64, Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x8B, 0x44, 0x24, 0x08]);
    }

    #[test]
    fn mov_rm_rbp_no_disp() {
        // mov rax, [rbp]  — RBP base with no disp: must use mod=01 + disp8=0
        // REX.W(48) + 8B + ModRM(mod01,reg0,rm5) + 00
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RBP), None, 1, 0);
        e.encode_mov_rm(OpSize::S64, Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x8B, 0x45, 0x00]);
    }

    #[test]
    fn mov_rm_full_sib() {
        // mov rax, [rbx + rcx*4 + 8]
        // REX.W(48) + 8B + ModRM(mod01,reg0,rm4=SIB) + SIB(ss2,idx1(rcx),base3(rbx)) + 08
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RBX), Some(Reg::RCX), 4, 8);
        e.encode_mov_rm(OpSize::S64, Reg::RAX, &addr);
        check(&e.buf, &[0x48, 0x8B, 0x44, 0x8B, 0x08]);
    }

    // All GPRs as MOV src and dst ─────────────────────────────────────────────

    #[test]
    fn mov_rr_all_gprs() {
        use Reg::*;
        let gprs = [
            RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
        ];
        for &dst in &gprs {
            for &src in &gprs {
                let mut e = enc();
                e.encode_mov_rr(OpSize::S64, dst, src);
                // Just verify it produces exactly 3 bytes (REX.W + opcode + ModRM)
                assert_eq!(
                    e.buf.len(),
                    3,
                    "mov {dst:?}, {src:?} should be 3 bytes, got {:02X?}",
                    e.buf
                );
                // REX.W must always be set
                assert_eq!(
                    e.buf[0] & 0xF8,
                    0x48,
                    "REX.W missing for mov {dst:?}, {src:?}"
                );
            }
        }
    }

    #[test]
    fn add_rr_all_gprs() {
        use Reg::*;
        let gprs = [
            RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
        ];
        for &dst in &gprs {
            for &src in &gprs {
                let mut e = enc();
                e.encode_add_rr(OpSize::S64, dst, src);
                assert_eq!(e.buf.len(), 3, "add {dst:?}, {src:?} should be 3 bytes");
                assert_eq!(
                    e.buf[0] & 0xF8,
                    0x48,
                    "REX.W missing for add {dst:?}, {src:?}"
                );
            }
        }
    }

    // encode_inst dispatch test ───────────────────────────────────────────────

    #[test]
    fn encode_inst_mov_rr() {
        let mut e = enc();
        let inst = MachInst::MovRR {
            size: OpSize::S64,
            dst: Operand::Reg(Reg::RAX),
            src: Operand::Reg(Reg::RCX),
        };
        e.encode_inst(&inst);
        check(&e.buf, &[0x48, 0x89, 0xC8]);
    }

    #[test]
    fn encode_inst_ret() {
        let mut e = enc();
        e.encode_inst(&MachInst::Ret);
        check(&e.buf, &[0xC3]);
    }

    // ── inst_size ────────────────────────────────────────────────────────────

    #[test]
    fn inst_size_mov_rr() {
        // REX.W + 89 + ModRM = 3 bytes
        let inst = MachInst::MovRR {
            size: OpSize::S64,
            dst: Operand::Reg(Reg::RAX),
            src: Operand::Reg(Reg::RCX),
        };
        assert_eq!(inst_size(&inst), 3);
    }

    #[test]
    fn inst_size_ret() {
        assert_eq!(inst_size(&MachInst::Ret), 1);
    }

    #[test]
    fn inst_size_jmp_near() {
        // Near JMP: E9 + rel32 = 5 bytes
        let inst = MachInst::Jmp { target: 0 };
        assert_eq!(inst_size(&inst), 5);
    }

    #[test]
    fn inst_size_jcc_near() {
        // Near Jcc: 0F 8x + rel32 = 6 bytes
        let inst = MachInst::Jcc {
            cc: CondCode::Eq,
            target: 0,
        };
        assert_eq!(inst_size(&inst), 6);
    }

    #[test]
    fn inst_size_add_ri_small() {
        // REX.W + 83 /0 + ib = 4 bytes
        let inst = MachInst::AddRI {
            size: OpSize::S64,
            dst: Operand::Reg(Reg::RAX),
            imm: 1,
        };
        assert_eq!(inst_size(&inst), 4);
    }

    // ── short-form jumps ─────────────────────────────────────────────────────

    #[test]
    fn jmp_short_encodes_eb() {
        // encode_jmp_short emits EB + one byte placeholder
        let mut e = enc();
        e.encode_jmp_short(0);
        // Before resolve: EB 00
        check(&e.buf, &[0xEB, 0x00]);
    }

    #[test]
    fn jcc_short_encodes_7x() {
        // encode_jcc_short(Eq) emits 74 + one byte placeholder (0x70 | 0x4 = 0x74)
        let mut e = enc();
        e.encode_jcc_short(CondCode::Eq, 0);
        check(&e.buf, &[0x74, 0x00]);
    }

    #[test]
    fn jmp_short_resolves_forward() {
        // Layout: [0] JMP_SHORT label; [1] RET; label here
        // EB offset=1; after emit: buf = [EB, 00, C3]
        // label bound at offset 3.
        // rel8 = target - (offset+1) = 3 - (1+1) = 1.
        let mut e = enc();
        e.encode_jmp_short(42);
        e.encode_ret(); // filler byte
        e.bind_label(42);
        e.resolve_fixups();
        check(&e.buf, &[0xEB, 0x01, 0xC3]);
    }

    #[test]
    fn jcc_short_resolves_forward() {
        // JE to label right after RET: buf[0]=74, buf[1]=00 (placeholder), buf[2]=C3; label at 3.
        // rel8 = 3 - (1+1) = 1
        let mut e = enc();
        e.encode_jcc_short(CondCode::Eq, 7);
        e.encode_ret();
        e.bind_label(7);
        e.resolve_fixups();
        check(&e.buf, &[0x74, 0x01, 0xC3]);
    }

    #[test]
    fn encode_inst_with_form_uses_short_for_jmp() {
        let mut e = enc();
        let inst = MachInst::Jmp { target: 0 };
        e.encode_inst_with_form(&inst, true);
        // Short JMP: EB + rel8 placeholder
        assert_eq!(e.buf[0], 0xEB);
        assert_eq!(e.buf.len(), 2);
    }

    #[test]
    fn encode_inst_with_form_uses_near_for_jmp() {
        let mut e = enc();
        let inst = MachInst::Jmp { target: 0 };
        e.encode_inst_with_form(&inst, false);
        // Near JMP: E9 + rel32
        assert_eq!(e.buf[0], 0xE9);
        assert_eq!(e.buf.len(), 5);
    }

    #[test]
    fn encode_inst_with_form_uses_short_for_jcc() {
        let mut e = enc();
        let inst = MachInst::Jcc {
            cc: CondCode::Ne,
            target: 0,
        };
        e.encode_inst_with_form(&inst, true);
        // Short JNE: 75 + rel8 placeholder
        assert_eq!(e.buf[0], 0x75);
        assert_eq!(e.buf.len(), 2);
    }

    #[test]
    fn encode_inst_with_form_uses_near_for_jcc() {
        let mut e = enc();
        let inst = MachInst::Jcc {
            cc: CondCode::Ne,
            target: 0,
        };
        e.encode_inst_with_form(&inst, false);
        // Near JNE: 0F 85 + rel32
        check(&e.buf[..2], &[0x0F, 0x85]);
        assert_eq!(e.buf.len(), 6);
    }

    // ── S32 encoding tests ──────────────────────────────────────────────────

    #[test]
    fn s32_add_eax_ecx() {
        // add eax, ecx -> no REX, 01, ModRM(C8)
        let mut e = enc();
        e.encode_add_rr(OpSize::S32, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x01, 0xC8]);
    }

    #[test]
    fn s32_add_r8d_r9d() {
        // add r8d, r9d -> REX(R+B = 0x45), 01, ModRM
        let mut e = enc();
        e.encode_add_rr(OpSize::S32, Reg::R8, Reg::R9);
        // REX: R=1(r9 hw=9>7), B=1(r8 hw=8>7) => 0x45
        // ModRM: mod=11, reg=1(r9&7), rm=0(r8&7) => 0xC8
        check(&e.buf, &[0x45, 0x01, 0xC8]);
    }

    #[test]
    fn s32_mov_eax_42() {
        // mov eax, 42 -> B8+0, imm32
        let mut e = enc();
        e.encode_mov_ri(OpSize::S32, Reg::RAX, 42);
        check(&e.buf, &[0xB8, 0x2A, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn s32_xor_eax_eax() {
        // xor eax, eax -> no REX, 31, C0
        let mut e = enc();
        e.encode_xor_rr(OpSize::S32, Reg::RAX, Reg::RAX);
        check(&e.buf, &[0x31, 0xC0]);
    }

    // ── S16 encoding tests ──────────────────────────────────────────────────

    #[test]
    fn s16_add_ax_cx() {
        // add ax, cx -> 66, 01, ModRM(C8)
        let mut e = enc();
        e.encode_add_rr(OpSize::S16, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x66, 0x01, 0xC8]);
    }

    #[test]
    fn s16_mov_ax_42() {
        // mov ax, 42 -> 66, B8+0, imm16
        let mut e = enc();
        e.encode_mov_ri(OpSize::S16, Reg::RAX, 42);
        check(&e.buf, &[0x66, 0xB8, 0x2A, 0x00]);
    }

    #[test]
    fn s16_sub_ax_cx() {
        // sub ax, cx -> 66, 29, ModRM(C8)
        let mut e = enc();
        e.encode_sub_rr(OpSize::S16, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x66, 0x29, 0xC8]);
    }

    #[test]
    fn s16_cmp_r8w_r9w() {
        // cmp r8w, r9w -> 66 REX(R+B) 39 ModRM
        let mut e = enc();
        e.encode_cmp_rr(OpSize::S16, Reg::R8, Reg::R9);
        // 0x66 prefix, REX: R=1(r9>7), B=1(r8>7) => 0x45
        // ModRM: mod=11, reg=1(r9&7), rm=0(r8&7) => 0xC8
        check(&e.buf, &[0x66, 0x45, 0x39, 0xC8]);
    }

    #[test]
    fn s16_neg_ax() {
        // neg ax -> 66, F7, /3 ModRM(D8)
        let mut e = enc();
        e.encode_neg(OpSize::S16, Reg::RAX);
        check(&e.buf, &[0x66, 0xF7, 0xD8]);
    }

    #[test]
    fn s16_inc_cx() {
        // inc cx -> 66, FF, /0 ModRM(C1)
        let mut e = enc();
        e.encode_inc(OpSize::S16, Reg::RCX);
        check(&e.buf, &[0x66, 0xFF, 0xC1]);
    }

    #[test]
    fn s16_shl_ax_3() {
        // shl ax, 3 -> 66, C1, /4 ModRM(E0), 03
        let mut e = enc();
        e.encode_shl_ri(OpSize::S16, Reg::RAX, 3);
        check(&e.buf, &[0x66, 0xC1, 0xE0, 0x03]);
    }

    #[test]
    fn s16_test_ax_cx() {
        // test ax, cx -> 66, 85, ModRM(C8)
        let mut e = enc();
        e.encode_test_rr(OpSize::S16, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x66, 0x85, 0xC8]);
    }

    #[test]
    fn s16_cmov_eq() {
        // cmove ax, cx -> 66, 0F 44, ModRM(C1)
        let mut e = enc();
        e.encode_cmov(OpSize::S16, CondCode::Eq, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x66, 0x0F, 0x44, 0xC1]);
    }

    #[test]
    fn s16_imul2_ax_cx() {
        // imul ax, cx -> 66, 0F AF, ModRM(C1)
        let mut e = enc();
        e.encode_imul2_rr(OpSize::S16, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x66, 0x0F, 0xAF, 0xC1]);
    }

    #[test]
    fn s16_mov_rr_ax_cx() {
        // mov ax, cx -> 66, 89, ModRM(C8)
        let mut e = enc();
        e.encode_mov_rr(OpSize::S16, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x66, 0x89, 0xC8]);
    }

    #[test]
    fn s16_alu_ri_small() {
        // add ax, 5 -> 66, 83, /0 ModRM(C0), 05
        let mut e = enc();
        e.encode_add_ri(OpSize::S16, Reg::RAX, 5);
        check(&e.buf, &[0x66, 0x83, 0xC0, 0x05]);
    }

    #[test]
    fn s16_alu_ri_large() {
        // add cx, 1000 -> 66, 81, /0 ModRM(C1), imm16
        let mut e = enc();
        e.encode_add_ri(OpSize::S16, Reg::RCX, 1000);
        check(&e.buf, &[0x66, 0x81, 0xC1, 0xE8, 0x03]);
    }

    #[test]
    fn s16_test_ri_rax() {
        // test ax, 0xFF -> 66, A9, imm16
        let mut e = enc();
        e.encode_test_ri(OpSize::S16, Reg::RAX, 0xFF);
        check(&e.buf, &[0x66, 0xA9, 0xFF, 0x00]);
    }

    #[test]
    fn s16_idiv_cx() {
        // idiv cx -> 66, F7, /7 ModRM(F9)
        let mut e = enc();
        e.encode_idiv(OpSize::S16, Reg::RCX);
        check(&e.buf, &[0x66, 0xF7, 0xF9]);
    }

    #[test]
    fn s16_dec_cx() {
        // dec cx -> 66, FF, /1 ModRM(C9)
        let mut e = enc();
        e.encode_dec(OpSize::S16, Reg::RCX);
        check(&e.buf, &[0x66, 0xFF, 0xC9]);
    }

    // ── S8 encoding tests ───────────────────────────────────────────────────

    #[test]
    fn s8_add_al_cl() {
        // add al, cl -> 00, ModRM(C8)
        // No REX needed: hw_enc(RAX)=0, hw_enc(RCX)=1, neither > 7 nor in 4-7
        let mut e = enc();
        e.encode_add_rr(OpSize::S8, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x00, 0xC8]);
    }

    #[test]
    fn s8_sub_al_dl() {
        // sub al, dl -> 28, ModRM(D0) -- SUB 0x29 & ~1 = 0x28
        let mut e = enc();
        e.encode_sub_rr(OpSize::S8, Reg::RAX, Reg::RDX);
        check(&e.buf, &[0x28, 0xD0]);
    }

    #[test]
    fn s8_add_spl_r8b() {
        // add spl, r8b -> REX(B), 00, ModRM
        // RSP hw_enc=4 (needs_byte_rex), R8 hw_enc=8 (>7)
        // REX: W=0, R=1(r8>7), B=0(rsp<=7) => 0x44
        let mut e = enc();
        e.encode_add_rr(OpSize::S8, Reg::RSP, Reg::R8);
        // ModRM: mod=11, reg=0(r8&7), rm=4(rsp&7) => 0xC4
        check(&e.buf, &[0x44, 0x00, 0xC4]);
    }

    #[test]
    fn s8_add_spl_bpl() {
        // add spl, bpl -> bare REX(0x40), 00, ModRM
        // RSP hw_enc=4, RBP hw_enc=5, both needs_byte_rex
        let mut e = enc();
        e.encode_add_rr(OpSize::S8, Reg::RSP, Reg::RBP);
        // REX: 0x40 (bare REX, no R/X/B bits)
        // ModRM: mod=11, reg=5(rbp&7), rm=4(rsp&7) => 0xEC
        check(&e.buf, &[0x40, 0x00, 0xEC]);
    }

    #[test]
    fn s8_mov_al_42() {
        // mov al, 42 -> B0, 2A
        let mut e = enc();
        e.encode_mov_ri(OpSize::S8, Reg::RAX, 42);
        check(&e.buf, &[0xB0, 0x2A]);
    }

    #[test]
    fn s8_mov_sil_42() {
        // mov sil, 42 -> REX(0x40), B0+6, 2A
        // RSI hw_enc=6 (needs_byte_rex)
        let mut e = enc();
        e.encode_mov_ri(OpSize::S8, Reg::RSI, 42);
        check(&e.buf, &[0x40, 0xB6, 0x2A]);
    }

    #[test]
    fn s8_mov_r8b_42() {
        // mov r8b, 42 -> REX.B(0x41), B0+0, 2A
        let mut e = enc();
        e.encode_mov_ri(OpSize::S8, Reg::R8, 42);
        check(&e.buf, &[0x41, 0xB0, 0x2A]);
    }

    #[test]
    fn s8_mov_rr_al_cl() {
        // mov al, cl -> 88, ModRM(C8)
        let mut e = enc();
        e.encode_mov_rr(OpSize::S8, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x88, 0xC8]);
    }

    #[test]
    fn s8_xor_al_al() {
        // xor al, al -> 30, C0 (XOR 0x31 & ~1 = 0x30)
        let mut e = enc();
        e.encode_xor_rr(OpSize::S8, Reg::RAX, Reg::RAX);
        check(&e.buf, &[0x30, 0xC0]);
    }

    #[test]
    fn s8_cmp_al_cl() {
        // cmp al, cl -> 38, ModRM(C8) (CMP 0x39 & ~1 = 0x38)
        let mut e = enc();
        e.encode_cmp_rr(OpSize::S8, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x38, 0xC8]);
    }

    #[test]
    fn s8_test_al_cl() {
        // test al, cl -> 84, ModRM(C8)
        let mut e = enc();
        e.encode_test_rr(OpSize::S8, Reg::RAX, Reg::RCX);
        check(&e.buf, &[0x84, 0xC8]);
    }

    #[test]
    fn s8_test_ri_al() {
        // test al, 0x42 -> F6 /0 ModRM(C0), 42
        let mut e = enc();
        e.encode_test_ri(OpSize::S8, Reg::RAX, 0x42);
        check(&e.buf, &[0xF6, 0xC0, 0x42]);
    }

    #[test]
    fn s8_neg_al() {
        // neg al -> F6 /3 ModRM(D8)
        let mut e = enc();
        e.encode_neg(OpSize::S8, Reg::RAX);
        check(&e.buf, &[0xF6, 0xD8]);
    }

    #[test]
    fn s8_not_al() {
        // not al -> F6 /2 ModRM(D0)
        let mut e = enc();
        e.encode_not(OpSize::S8, Reg::RAX);
        check(&e.buf, &[0xF6, 0xD0]);
    }

    #[test]
    fn s8_inc_al() {
        // inc al -> FE /0 ModRM(C0)
        let mut e = enc();
        e.encode_inc(OpSize::S8, Reg::RAX);
        check(&e.buf, &[0xFE, 0xC0]);
    }

    #[test]
    fn s8_dec_cl() {
        // dec cl -> FE /1 ModRM(C9)
        let mut e = enc();
        e.encode_dec(OpSize::S8, Reg::RCX);
        check(&e.buf, &[0xFE, 0xC9]);
    }

    #[test]
    fn s8_shl_al_3() {
        // shl al, 3 -> C0 /4 ModRM(E0), 03
        let mut e = enc();
        e.encode_shl_ri(OpSize::S8, Reg::RAX, 3);
        check(&e.buf, &[0xC0, 0xE0, 0x03]);
    }

    #[test]
    fn s8_shr_al_1() {
        // shr al, 1 -> D0 /5 ModRM(E8)
        let mut e = enc();
        e.encode_shr_ri(OpSize::S8, Reg::RAX, 1);
        check(&e.buf, &[0xD0, 0xE8]);
    }

    #[test]
    fn s8_sar_cl_shift() {
        // sar cl, cl -> D2 /7 ModRM(F9)
        let mut e = enc();
        e.encode_sar_rcl(OpSize::S8, Reg::RCX);
        check(&e.buf, &[0xD2, 0xF9]);
    }

    #[test]
    fn s8_alu_ri_add() {
        // add al, 5 -> 80 /0 ModRM(C0), 05
        let mut e = enc();
        e.encode_add_ri(OpSize::S8, Reg::RAX, 5);
        check(&e.buf, &[0x80, 0xC0, 0x05]);
    }

    #[test]
    fn s8_idiv_cl() {
        // idiv cl -> F6 /7 ModRM(F9)
        let mut e = enc();
        e.encode_idiv(OpSize::S8, Reg::RCX);
        check(&e.buf, &[0xF6, 0xF9]);
    }

    #[test]
    fn s8_div_cl() {
        // div cl -> F6 /6 ModRM(F1)
        let mut e = enc();
        e.encode_div(OpSize::S8, Reg::RCX);
        check(&e.buf, &[0xF6, 0xF1]);
    }

    #[test]
    #[should_panic(expected = "IMUL has no byte form")]
    fn s8_imul2_panics() {
        let mut e = enc();
        e.encode_imul2_rr(OpSize::S8, Reg::RAX, Reg::RCX);
    }

    #[test]
    #[should_panic(expected = "IMUL has no byte form")]
    fn s8_imul3_panics() {
        let mut e = enc();
        e.encode_imul3_rri(OpSize::S8, Reg::RAX, Reg::RCX, 5);
    }

    #[test]
    #[should_panic(expected = "CMOV has no byte form")]
    fn s8_cmov_panics() {
        let mut e = enc();
        e.encode_cmov(OpSize::S8, CondCode::Eq, Reg::RAX, Reg::RCX);
    }

    #[test]
    #[should_panic(expected = "LEA has no byte form")]
    fn s8_lea_panics() {
        let mut e = enc();
        let addr = Addr::new(Some(Reg::RAX), None, 1, 0);
        e.encode_lea(OpSize::S8, Reg::RAX, &addr);
    }

    #[test]
    fn s8_inc_spl_needs_rex() {
        // inc spl -> REX(0x40), FE /0 ModRM(C4)
        let mut e = enc();
        e.encode_inc(OpSize::S8, Reg::RSP);
        check(&e.buf, &[0x40, 0xFE, 0xC4]);
    }

    #[test]
    fn s8_neg_dil_needs_rex() {
        // neg dil -> REX(0x40), F6 /3 ModRM(DF)
        let mut e = enc();
        e.encode_neg(OpSize::S8, Reg::RDI);
        // ModRM: mod=11, /3=3, rm=7(rdi) => 0b11_011_111 = 0xDF
        check(&e.buf, &[0x40, 0xF6, 0xDF]);
    }

    #[test]
    fn s8_or_al_bl() {
        // or al, bl -> 08, ModRM(D8) (OR 0x09 & ~1 = 0x08)
        let mut e = enc();
        e.encode_or_rr(OpSize::S8, Reg::RAX, Reg::RBX);
        // ModRM: mod=11, reg=3(rbx), rm=0(rax) => 0xD8
        check(&e.buf, &[0x08, 0xD8]);
    }

    #[test]
    fn s8_and_al_dl() {
        // and al, dl -> 20, ModRM(D0) (AND 0x21 & ~1 = 0x20)
        let mut e = enc();
        e.encode_and_rr(OpSize::S8, Reg::RAX, Reg::RDX);
        check(&e.buf, &[0x20, 0xD0]);
    }
}
