use super::*;

impl Encoder {
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

    /// Emit SSE conversion with REX.W prefix: mandatory_prefix + REX.W + 0F + opcode + ModRM.
    fn encode_sse_rr_w(&mut self, prefix: u8, opcode: u8, dst: Reg, src: Reg) {
        let d = dst.hw_enc();
        let s = src.hw_enc();
        self.emit_byte(prefix);
        self.emit_rex(true, d, 0, s);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_modrm(0b11, d, s);
    }

    /// cvtsi2sd: F2 0F 2A /r (GPR -> XMM, int -> f64)
    pub fn encode_cvtsi2sd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr_w(0xF2, 0x2A, dst, src);
    }

    /// cvtsi2ss: F3 0F 2A /r (GPR -> XMM, int -> f32)
    pub fn encode_cvtsi2ss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr_w(0xF3, 0x2A, dst, src);
    }

    /// cvttsd2si: F2 0F 2C /r (XMM -> GPR, f64 -> int truncation)
    pub fn encode_cvttsd2si_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr_w(0xF2, 0x2C, dst, src);
    }

    /// cvttss2si: F3 0F 2C /r (XMM -> GPR, f32 -> int truncation)
    pub fn encode_cvttss2si_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr_w(0xF3, 0x2C, dst, src);
    }

    /// cvtsd2ss: F2 0F 5A /r (XMM -> XMM, f64 -> f32)
    pub fn encode_cvtsd2ss_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF2, 0x5A, dst, src);
    }

    /// cvtss2sd: F3 0F 5A /r (XMM -> XMM, f32 -> f64)
    pub fn encode_cvtss2sd_rr(&mut self, dst: Reg, src: Reg) {
        self.encode_sse_rr(0xF3, 0x5A, dst, src);
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
}
