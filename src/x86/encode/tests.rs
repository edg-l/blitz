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
