use super::*;

impl Encoder {
    // ── Instruction encoders ──────────────────────────────────────────────

    /// MOV r/m, r  — used as the canonical RR form.
    pub fn encode_mov_rr(&mut self, size: OpSize, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_prefix_and_rex(size, s, 0, d);
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
        self.emit_prefix_and_rex(size, d, idx, base);
        let opcode = if size == OpSize::S8 { 0x8A } else { 0x8B };
        self.emit_byte(opcode);
        self.emit_addr(d, addr);
    }

    /// MOV r/m, r  (store to memory).
    pub fn encode_mov_mr(&mut self, size: OpSize, addr: &Addr, src: Reg) {
        let s = src.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_prefix_and_rex(size, s, idx, base);
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
        self.emit_prefix_and_rex(size, s, 0, d);
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
        self.emit_prefix_and_rex(size, d, idx, base);
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
        self.emit_prefix_and_rex(size, s, 0, d);
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
        self.emit_prefix_and_rex(size, 0, 0, d);
        if imm == 1 {
            let opcode = if size == OpSize::S8 { 0xD0 } else { 0xD1 };
            self.emit_byte(opcode);
            self.emit_modrm(0b11, slash_n, d);
        } else {
            let opcode = if size == OpSize::S8 { 0xC0 } else { 0xC1 };
            self.emit_byte(opcode);
            self.emit_modrm(0b11, slash_n, d);
            self.emit_byte(imm);
        }
    }

    fn encode_shift_cl(&mut self, size: OpSize, slash_n: u8, dst: Reg) {
        let d = dst.hw_enc();
        self.emit_prefix_and_rex(size, 0, 0, d);
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
        self.emit_prefix_and_rex(size, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0xAF);
        self.emit_modrm(0b11, d, s);
    }

    pub fn encode_imul3_rri(&mut self, size: OpSize, dst: Reg, src: Reg, imm: i32) {
        assert!(size != OpSize::S8, "IMUL has no byte form");
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_prefix_and_rex(size, d, 0, s);
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

    // ── CDQ / CQO / IDIV / DIV / NEG / NOT ───────────────────────────────

    pub fn encode_cdq(&mut self) {
        self.emit_byte(0x99);
    }

    pub fn encode_cqo(&mut self) {
        self.emit_rex(true, 0, 0, 0);
        self.emit_byte(0x99);
    }

    /// CWD: sign-extend AX to DX:AX (0x66 0x99).
    pub fn encode_cwd(&mut self) {
        self.emit_size_prefix(OpSize::S16);
        self.emit_byte(0x99);
    }

    /// CBW: sign-extend AL to AX (0x66 0x98).
    ///
    /// In 64-bit mode, bare opcode 0x98 is CWDE (sign-extend EAX to RAX).
    /// The 0x66 operand-size override prefix selects the 16-bit form (CBW),
    /// which sign-extends AL into AX as needed for 8-bit division setup.
    pub fn encode_cbw(&mut self) {
        self.emit_size_prefix(OpSize::S16);
        self.emit_byte(0x98);
    }

    fn encode_unary_group(&mut self, size: OpSize, slash_n: u8, src: Reg) {
        let s = src.hw_enc();
        self.emit_prefix_and_rex(size, 0, 0, s);
        let opcode = if size == OpSize::S8 { 0xF6 } else { 0xF7 };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, slash_n, s);
    }

    fn encode_incdec(&mut self, size: OpSize, slash_n: u8, dst: Reg) {
        let d = dst.hw_enc();
        self.emit_prefix_and_rex(size, 0, 0, d);
        let opcode = if size == OpSize::S8 { 0xFE } else { 0xFF };
        self.emit_byte(opcode);
        self.emit_modrm(0b11, slash_n, d);
    }

    pub fn encode_idiv(&mut self, size: OpSize, src: Reg) {
        self.encode_unary_group(size, 7, src);
    }

    pub fn encode_div(&mut self, size: OpSize, src: Reg) {
        self.encode_unary_group(size, 6, src);
    }

    pub fn encode_neg(&mut self, size: OpSize, dst: Reg) {
        self.encode_unary_group(size, 3, dst);
    }

    pub fn encode_not(&mut self, size: OpSize, dst: Reg) {
        self.encode_unary_group(size, 2, dst);
    }

    /// INC r/m
    pub fn encode_inc(&mut self, size: OpSize, dst: Reg) {
        self.encode_incdec(size, 0, dst);
    }

    /// DEC r/m
    pub fn encode_dec(&mut self, size: OpSize, dst: Reg) {
        self.encode_incdec(size, 1, dst);
    }

    // ── MOVZX / MOVSX ─────────────────────────────────────────────────────

    /// Emit REX.W + 0F + opcode + ModRM for reg-reg zero/sign-extend.
    fn encode_0f_rr(&mut self, opcode: u8, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_modrm(0b11, d, s);
    }

    pub fn encode_movzx_br(&mut self, dst: Reg, src: Reg) {
        self.encode_0f_rr(0xB6, dst, src);
    }

    pub fn encode_movsx_br(&mut self, dst: Reg, src: Reg) {
        self.encode_0f_rr(0xBE, dst, src);
    }

    pub fn encode_movzx_wr(&mut self, dst: Reg, src: Reg) {
        self.encode_0f_rr(0xB7, dst, src);
    }

    pub fn encode_movsx_wr(&mut self, dst: Reg, src: Reg) {
        self.encode_0f_rr(0xBF, dst, src);
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

    pub fn encode_movsx_dr(&mut self, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x63);
        self.emit_modrm(0b11, d, s);
    }
}
