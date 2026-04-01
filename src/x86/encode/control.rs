use super::*;

impl Encoder {
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
