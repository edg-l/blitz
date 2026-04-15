use super::*;

impl Encoder {
    // ── LEA ───────────────────────────────────────────────────────────────

    pub fn encode_lea(&mut self, size: OpSize, dst: Reg, addr: &Addr) {
        assert!(size != OpSize::S8, "LEA has no byte form");
        let d = dst.hw_enc();
        let idx = addr.index.map_or(0u8, |r| r.hw_enc());
        let base = addr.base.map_or(0u8, |r| r.hw_enc());
        self.emit_prefix_and_rex(size, d, idx, base);
        self.emit_byte(0x8D);
        self.emit_addr(d, addr);
    }

    // ── LEA RIP-relative ───────────────────────────────────────────────────

    /// LEA dst, [RIP + disp32] with a PC32 relocation for `symbol`.
    /// Encoding: REX.W 8D /r (mod=00, rm=5 = RIP-relative).
    pub fn encode_lea_rip_relative(&mut self, dst: Reg, symbol: &str) {
        let d = dst.hw_enc();
        // REX.W prefix (64-bit operand)
        self.emit_rex(true, d, 0, 0);
        // Opcode: LEA
        self.emit_byte(0x8D);
        // ModRM: mod=00, reg=dst, rm=5 (RIP-relative addressing)
        self.emit_modrm(0b00, d, 5);
        // Record relocation offset (where the disp32 placeholder starts)
        let offset = self.buf.len();
        // disp32 placeholder
        self.emit_le32(0);
        self.relocations.push(Reloc {
            offset,
            kind: RelocKind::PC32,
            symbol: symbol.to_string(),
            addend: -4,
        });
    }

    // ── PUSH / POP ────────────────────────────────────────────────────────

    pub fn encode_push(&mut self, src: Reg) {
        let s = src.hw_enc();
        self.maybe_emit_rex(false, 0, 0, s);
        self.emit_byte(0x50 | (s & 7));
    }

    pub fn encode_pop(&mut self, dst: Reg) {
        let d = dst.hw_enc();
        self.maybe_emit_rex(false, 0, 0, d);
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

    // ── SETCC / CMOV ──────────────────────────────────────────────────────

    pub fn encode_setcc(&mut self, cc: CondCode, dst: Reg) {
        let tttn = Self::cc_byte(cc);
        let d = dst.hw_enc();
        self.emit_rex_for_size(OpSize::S8, 0, 0, d);
        self.emit_byte(0x0F);
        self.emit_byte(0x90 | tttn);
        self.emit_modrm(0b11, 0, d);
    }

    pub fn encode_cmov(&mut self, size: OpSize, cc: CondCode, dst: Reg, src: Reg) {
        assert!(size != OpSize::S8, "CMOV has no byte form");
        let tttn = Self::cc_byte(cc);
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_prefix_and_rex(size, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(0x40 | tttn);
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
}
