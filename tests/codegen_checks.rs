//! Builder-based codegen lit tests.
//!
//! Each test constructs IR with FunctionBuilder, compiles it, and runs
//! CHECK/CHECK-NEXT/CHECK-NOT patterns against either the disassembly
//! (asm tests) or the optimized IR text (ir tests).
//!
//! This is the Blitz equivalent of LLVM's FileCheck workflow.

use blitz::compile::{CompileOptions, compile, compile_to_ir_string};
use blitz::ir::builder::FunctionBuilder;
use blitz::ir::condcode::CondCode;
use blitz::ir::function::Function;
use blitz::ir::types::Type;
use blitz::test_utils::objdump_disasm;
use blitztest::check::run_checks;
use blitztest::directive::{CheckPattern, Directive, parse_directives};

fn extract_checks(checks: &str) -> Vec<CheckPattern> {
    let directives = parse_directives(checks).expect("bad CHECK directives");
    directives
        .into_iter()
        .filter_map(|d| match d {
            Directive::Check(p) => Some(p),
            _ => None,
        })
        .collect()
}

/// Compile a function and run CHECK directives against its disassembly.
fn check_asm(func: Function, checks: &str) {
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile failed");
    let disasm = objdump_disasm(&obj.code).expect("objdump not available");

    let owned = extract_checks(checks);
    let patterns: Vec<&CheckPattern> = owned.iter().collect();

    if let Err(e) = run_checks(&disasm, &patterns) {
        panic!("CHECK failure: {e}\n\n--- disassembly ---\n{disasm}\n--- checks ---\n{checks}");
    }
}

/// Compile a function to IR and run CHECK directives against the IR text.
fn check_ir(func: Function, checks: &str) {
    let opts = CompileOptions::default();
    let ir = compile_to_ir_string(func, &opts).expect("compile_to_ir_string failed");

    let owned = extract_checks(checks);
    let patterns: Vec<&CheckPattern> = owned.iter().collect();

    if let Err(e) = run_checks(&ir, &patterns) {
        panic!("CHECK failure: {e}\n\n--- IR ---\n{ir}\n--- checks ---\n{checks}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IR-level tests: verify e-graph optimization and isel transforms
// ═══════════════════════════════════════════════════════════════════════════════

/// mul(x, 3) is rewritten to addr(scale=2) in isel (lea [x+x*2])
#[test]
fn ir_mul3_to_lea() {
    let mut b = FunctionBuilder::new("sr_mul3", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c3 = b.iconst(3, Type::I64);
    let r = b.mul(x, c3);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sr_mul3
        // CHECK: v0 = param(0, I64)
        // CHECK-NEXT: v1 = addr(scale=2, disp=0)(v0, v0)
        // CHECK: ret v1
        ",
    );
}

/// Chained strength reductions: mul->shl, udiv->shr, then addr fusion
#[test]
fn ir_chained_sr() {
    let mut b = FunctionBuilder::new("sr_chain", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let c4 = b.iconst(4, Type::I64);
    let c16 = b.iconst(16, Type::I64);
    let a = b.mul(params[0], c4);
    let d = b.udiv(params[1], c16);
    let r = b.add(a, d);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sr_chain
        // CHECK: v0 = param(0, I64)
        // CHECK-NEXT: v1 = x86_shl_imm(2)(v0)
        // CHECK-NEXT: v2 = proj0(v1)
        // CHECK-NEXT: v3 = param(1, I64)
        // CHECK-NEXT: v4 = x86_shr_imm(4)(v3)
        // CHECK-NEXT: v5 = proj0(v4)
        // CHECK-NEXT: v6 = addr(scale=1, disp=0)(v2, v5)
        // CHECK: ret v6
        ",
    );
}

/// Diamond CFG: branch, two blocks with addr ops, merge via block param
#[test]
fn ir_diamond_cfg() {
    let mut b = FunctionBuilder::new("diamond", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let bb_true = b.create_block();
    let bb_false = b.create_block();
    let (bb_exit, exit_params) = b.create_block_with_params(&[Type::I64]);
    let cond = b.icmp(CondCode::Sgt, params[0], params[1]);
    b.branch(cond, bb_true, bb_false, &[], &[]);
    b.set_block(bb_true);
    let c1a = b.iconst(1, Type::I64);
    let ta = b.add(params[0], c1a);
    b.jump(bb_exit, &[ta]);
    b.set_block(bb_false);
    let c1b = b.iconst(1, Type::I64);
    let fb = b.add(params[1], c1b);
    b.jump(bb_exit, &[fb]);
    b.set_block(bb_exit);
    b.ret(Some(exit_params[0]));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function diamond
        // CHECK: v0 = param(0, I64)
        // CHECK-NEXT: v1 = param(1, I64)
        // CHECK-NEXT: v2 = x86_sub(v0, v1)
        // CHECK-NEXT: v3 = proj1(v2)
        // CHECK: branch Sgt v3 block1() block2()
        // CHECK-LABEL: block1:
        // CHECK: v6 = addr(scale=1, disp=0)(v0, v4)
        // CHECK: jump block3
        // CHECK-LABEL: block2:
        // CHECK: v4 = iconst(1, I64)
        // CHECK-NEXT: v5 = addr(scale=1, disp=0)(v1, v4)
        // CHECK: jump block3
        // CHECK-LABEL: block3(p0: I64):
        // CHECK: v7 = block_param(b3, 0, I64)
        // CHECK: ret v7
        ",
    );
}

/// Counted loop: xor init, addr increments, sub for compare, back-edge branch
#[test]
fn ir_counted_loop() {
    let mut b = FunctionBuilder::new("loop_sum", &[Type::I64], &[Type::I64]);
    let n = b.params().to_vec()[0];
    let (bb_loop, lp) = b.create_block_with_params(&[Type::I64, Type::I64]);
    let (bb_exit, ep) = b.create_block_with_params(&[Type::I64]);
    let zero = b.iconst(0, Type::I64);
    b.jump(bb_loop, &[zero, zero]);
    b.set_block(bb_loop);
    let new_acc = b.add(lp[0], lp[1]);
    let one = b.iconst(1, Type::I64);
    let new_i = b.add(lp[1], one);
    let cond = b.icmp(CondCode::Slt, new_i, n);
    b.branch(cond, bb_loop, bb_exit, &[new_acc, new_i], &[new_acc]);
    b.set_block(bb_exit);
    b.ret(Some(ep[0]));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function loop_sum
        // CHECK: v0 = iconst(0, I64)
        // CHECK: jump block3(v0, v0)
        // CHECK-LABEL: block1(p0: I64, p1: I64):
        // CHECK: addr(scale=1, disp=0)
        // CHECK: x86_sub
        // CHECK: proj1
        // CHECK: addr(scale=1, disp=0)
        // CHECK: branch Slt
        // CHECK-LABEL: block2(p0: I64):
        // CHECK: ret
        // CHECK-LABEL: block3(p0: I64, p1: I64):
        // CHECK: param(0, I64)
        // CHECK: iconst(1, I64)
        // CHECK: jump block1
        ",
    );
}

/// Constant folding: (5*3)+(10-2))/1 fully folds to iconst(23)
#[test]
fn ir_constfold_to_iconst() {
    let mut b = FunctionBuilder::new("cf_complex", &[], &[Type::I64]);
    let c5 = b.iconst(5, Type::I64);
    let c3 = b.iconst(3, Type::I64);
    let c10 = b.iconst(10, Type::I64);
    let c2 = b.iconst(2, Type::I64);
    let c1 = b.iconst(1, Type::I64);
    let a = b.mul(c5, c3);
    let d = b.sub(c10, c2);
    let sum = b.add(a, d);
    let r = b.udiv(sum, c1);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function cf_complex
        // CHECK: v0 = iconst(23, I64)
        // CHECK-NOT: mul
        // CHECK-NOT: sub
        // CHECK-NOT: add
        // CHECK-NOT: udiv
        // CHECK: ret v0
        ",
    );
}

/// Dead code elimination: unused mul(x,y) doesn't appear in IR
#[test]
fn ir_dce_removes_unused() {
    let mut b = FunctionBuilder::new("dce", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let _dead = b.mul(params[0], params[1]);
    let r = b.add(params[0], params[1]);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function dce
        // CHECK: v0 = param(0, I64)
        // CHECK-NEXT: v1 = param(1, I64)
        // CHECK-NEXT: v2 = addr(scale=1, disp=0)(v0, v1)
        // CHECK-NOT: mul
        // CHECK: ret v2
        ",
    );
}

/// urem(x, 8) -> and(x, 7) in the IR
#[test]
fn ir_urem_pow2_to_and() {
    let mut b = FunctionBuilder::new("urem8", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c8 = b.iconst(8, Type::I64);
    let r = b.urem(x, c8);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function urem8
        // CHECK: v0 = param(0, I64)
        // CHECK-NEXT: v1 = iconst(7, I64)
        // CHECK-NEXT: v2 = x86_and(v0, v1)
        // CHECK-NEXT: v3 = proj0(v2)
        // CHECK-NOT: urem
        // CHECK-NOT: div
        // CHECK: ret v3
        ",
    );
}

/// Nested max(max(a,b), c) uses x86_sub + x86_cmov pairs
#[test]
fn ir_nested_select_to_cmov() {
    let mut b = FunctionBuilder::new("max3", &[Type::I64, Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let cond1 = b.icmp(CondCode::Sgt, params[0], params[1]);
    let m1 = b.select(cond1, params[0], params[1]);
    let cond2 = b.icmp(CondCode::Sgt, m1, params[2]);
    let r = b.select(cond2, m1, params[2]);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function max3
        // CHECK: v0 = param(0, I64)
        // CHECK-NEXT: v1 = param(1, I64)
        // CHECK-NEXT: v2 = x86_sub(v0, v1)
        // CHECK-NEXT: v3 = proj1(v2)
        // CHECK-NEXT: v4 = x86_cmov(Sgt)(v3, v0, v1)
        // CHECK-NEXT: v5 = param(2, I64)
        // CHECK-NEXT: v6 = x86_sub(v4, v5)
        // CHECK-NEXT: v7 = proj1(v6)
        // CHECK-NEXT: v8 = x86_cmov(Sgt)(v7, v4, v5)
        // CHECK: ret v8
        ",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASM-level tests: verify final x86-64 instruction sequences
// ═══════════════════════════════════════════════════════════════════════════════

/// mul(x, 3) -> single lea [rdi+rdi*2], ret
#[test]
fn asm_mul3_to_lea() {
    let mut b = FunctionBuilder::new("sr_mul3", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c3 = b.iconst(3, Type::I64);
    let r = b.mul(x, c3);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: lea    rax,[rdi+rdi*2]
        // CHECK-NEXT: ret
        ",
    );
}

/// mul(x, 5) -> single lea [rdi+rdi*4], ret
#[test]
fn asm_mul5_to_lea() {
    let mut b = FunctionBuilder::new("sr_mul5", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c5 = b.iconst(5, Type::I64);
    let r = b.mul(x, c5);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: lea    rax,[rdi+rdi*4]
        // CHECK-NEXT: ret
        ",
    );
}

/// mul(x, 8) -> mov + shl 3 + mov + ret (no imul)
#[test]
fn asm_mul_pow2_to_shl() {
    let mut b = FunctionBuilder::new("sr_mul8", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c8 = b.iconst(8, Type::I64);
    let r = b.mul(x, c8);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    rcx,rdi
        // CHECK-NEXT: shl    rcx,0x3
        // CHECK-NEXT: mov    rax,rcx
        // CHECK-NEXT: ret
        ",
    );
}

/// Chained: shl(x,2) + shr(y,4) + lea, no imul/div
#[test]
fn asm_chained_sr() {
    let mut b = FunctionBuilder::new("sr_chain", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let c4 = b.iconst(4, Type::I64);
    let c16 = b.iconst(16, Type::I64);
    let a = b.mul(params[0], c4);
    let d = b.udiv(params[1], c16);
    let r = b.add(a, d);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: shl    rax,0x2
        // CHECK-NOT: imul
        // CHECK: shr    rax,0x4
        // CHECK-NOT: div
        // CHECK: lea    rax,[rdx+rcx*1]
        // CHECK-NEXT: ret
        ",
    );
}

/// add(base, shl(idx, 2)) -> single lea [rdi+rsi*4], ret
#[test]
fn asm_addr_mode_scaled() {
    let mut b = FunctionBuilder::new("addr_scale", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let c2 = b.iconst(2, Type::I64);
    let scaled = b.shl(params[1], c2);
    let r = b.add(params[0], scaled);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: lea    rax,[rdi+rsi*4]
        // CHECK-NEXT: ret
        ",
    );
}

/// base+idx*8+16 -> lea [rdi+rsi*8] then lea [rcx+0x10]
#[test]
fn asm_addr_mode_full() {
    let mut b = FunctionBuilder::new("addr_full", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let c3 = b.iconst(3, Type::I64);
    let c16 = b.iconst(16, Type::I64);
    let scaled = b.shl(params[1], c3);
    let sum = b.add(params[0], scaled);
    let r = b.add(sum, c16);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: lea    rcx,[rdi+rsi*8]
        // CHECK: mov    rdx,0x10
        // CHECK-NEXT: lea    rax,[rdx+rcx*1]
        // CHECK-NOT: shl
        // CHECK-NEXT: ret
        ",
    );
}

/// sub sets flags, cmov consumes them, no separate cmp
#[test]
fn asm_flag_fusion_sub_cmov() {
    let mut b = FunctionBuilder::new("flag_fuse", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let diff = b.sub(params[0], params[1]);
    let zero = b.iconst(0, Type::I64);
    let cond = b.icmp(CondCode::Sgt, diff, zero);
    let r = b.select(cond, diff, zero);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    rax,rdi
        // CHECK-NEXT: sub    rax,rsi
        // The flags-only compare against 0 is `test r, r` (X86CmpI(0)).
        // CHECK: test   {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: cmovg  rax,{{[a-z0-9]+}}
        // CHECK-NEXT: ret
        ",
    );
}

/// (3+7)*2-4 fully folds to mov rax,0x10; ret
#[test]
fn asm_constfold_chain() {
    let mut b = FunctionBuilder::new("cf_chain", &[], &[Type::I64]);
    let c3 = b.iconst(3, Type::I64);
    let c7 = b.iconst(7, Type::I64);
    let c2 = b.iconst(2, Type::I64);
    let c4 = b.iconst(4, Type::I64);
    let sum = b.add(c3, c7);
    let prod = b.mul(sum, c2);
    let r = b.sub(prod, c4);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    rax,0x10
        // CHECK-NEXT: ret
        ",
    );
}

/// shl(1, 10) -> mov rax,0x400; ret
#[test]
fn asm_constfold_shift() {
    let mut b = FunctionBuilder::new("cf_shift", &[], &[Type::I64]);
    let c1 = b.iconst(1, Type::I64);
    let c10 = b.iconst(10, Type::I64);
    let r = b.shl(c1, c10);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    rax,0x400
        // CHECK-NEXT: ret
        ",
    );
}

/// 4-arg call sets up rdi, rsi, rdx, rcx in SysV ABI order
#[test]
fn asm_call_4arg_abi() {
    let mut b = FunctionBuilder::new("caller4", &[], &[Type::I64]);
    let c1 = b.iconst(1, Type::I64);
    let c2 = b.iconst(2, Type::I64);
    let c3 = b.iconst(3, Type::I64);
    let c4 = b.iconst(4, Type::I64);
    let ret = b.call("ext_fn", &[c1, c2, c3, c4], &[Type::I64]);
    b.ret(Some(ret[0]));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    {{[a-z0-9]+}},0x1
        // CHECK: mov    {{[a-z0-9]+}},0x2
        // CHECK: mov    {{[a-z0-9]+}},0x3
        // CHECK: mov    {{[a-z0-9]+}},0x4
        // CHECK: call
        // CHECK: ret
        ",
    );
}

/// Diamond branch: sub for compare, jg, two paths with lea, ret
#[test]
fn asm_diamond_branch() {
    let mut b = FunctionBuilder::new("diamond", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let bb_true = b.create_block();
    let bb_false = b.create_block();
    let (bb_exit, exit_params) = b.create_block_with_params(&[Type::I64]);
    let cond = b.icmp(CondCode::Sgt, params[0], params[1]);
    b.branch(cond, bb_true, bb_false, &[], &[]);
    b.set_block(bb_true);
    let c1a = b.iconst(1, Type::I64);
    let ta = b.add(params[0], c1a);
    b.jump(bb_exit, &[ta]);
    b.set_block(bb_false);
    let c1b = b.iconst(1, Type::I64);
    let fb = b.add(params[1], c1b);
    b.jump(bb_exit, &[fb]);
    b.set_block(bb_exit);
    b.ret(Some(exit_params[0]));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: jg
        // CHECK: mov    {{[a-z0-9]+}},0x1
        // CHECK: lea    {{[a-z0-9]+}},
        // CHECK: jmp
        // CHECK: mov    {{[a-z0-9]+}},0x1
        // CHECK: lea    {{[a-z0-9]+}},
        // CHECK: mov    rax,
        // CHECK-NEXT: ret
        ",
    );
}

/// Loop: xor init, lea for increment, sub for compare, jge + jmp back-edge
#[test]
fn asm_counted_loop() {
    let mut b = FunctionBuilder::new("loop_sum", &[Type::I64], &[Type::I64]);
    let n = b.params().to_vec()[0];
    let (bb_loop, lp) = b.create_block_with_params(&[Type::I64, Type::I64]);
    let (bb_exit, ep) = b.create_block_with_params(&[Type::I64]);
    let zero = b.iconst(0, Type::I64);
    b.jump(bb_loop, &[zero, zero]);
    b.set_block(bb_loop);
    let new_acc = b.add(lp[0], lp[1]);
    let one = b.iconst(1, Type::I64);
    let new_i = b.add(lp[1], one);
    let cond = b.icmp(CondCode::Slt, new_i, n);
    b.branch(cond, bb_loop, bb_exit, &[new_acc, new_i], &[new_acc]);
    b.set_block(bb_exit);
    b.ret(Some(ep[0]));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: xor    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: lea    {{[a-z0-9]+}},
        // CHECK: lea    {{[a-z0-9]+}},
        // CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK-NEXT: jge
        // CHECK: jmp
        // CHECK: ret
        ",
    );
}

/// I32 add uses 32-bit registers throughout
#[test]
fn asm_i32_uses_32bit_regs() {
    let mut b = FunctionBuilder::new("add_i32", &[Type::I32, Type::I32], &[Type::I32]);
    let params = b.params().to_vec();
    let r = b.add(params[0], params[1]);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    ecx,edi
        // CHECK-NEXT: add    ecx,esi
        // CHECK-NEXT: mov    eax,ecx
        // CHECK-NEXT: ret
        ",
    );
}

/// Unused mul is dead-code-eliminated; only lea for the add survives
#[test]
fn asm_dce_eliminates_unused() {
    let mut b = FunctionBuilder::new("dce", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let _dead = b.mul(params[0], params[1]);
    let r = b.add(params[0], params[1]);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: lea    rax,[rdi+rsi*1]
        // CHECK-NEXT: ret
        ",
    );
}

/// iconst(0) -> xor rax,rax (2-byte idiom, not 10-byte mov)
#[test]
fn asm_zero_via_xor() {
    let mut b = FunctionBuilder::new("ret_zero", &[], &[Type::I64]);
    let c0 = b.iconst(0, Type::I64);
    b.ret(Some(c0));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: xor    rax,rax
        // CHECK-NEXT: ret
        ",
    );
}

/// sub(x, x) algebraically folds to 0, emits xor (no sub in output)
#[test]
fn asm_sub_self_to_xor() {
    let mut b = FunctionBuilder::new("sub_self", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let r = b.sub(x, x);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: xor    rax,rax
        // CHECK-NOT: sub
        // CHECK-NEXT: ret
        ",
    );
}

/// sext(i32, i64) -> movsxd rax,edi; ret
#[test]
fn asm_sext_i32_to_i64() {
    let mut b = FunctionBuilder::new("sext32", &[Type::I32], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let r = b.sext(x, Type::I64);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: movsxd rax,edi
        // CHECK-NEXT: ret
        ",
    );
}

/// zext(i8, i64) -> movzx rax,dil; ret
#[test]
fn asm_zext_i8_to_i64() {
    let mut b = FunctionBuilder::new("zext8", &[Type::I8], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let r = b.zext(x, Type::I64);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: movzx  rax,dil
        // CHECK-NEXT: ret
        ",
    );
}

/// urem(x, 8) -> and with 0x7 mask, no div instruction
#[test]
fn asm_urem_pow2_to_and() {
    let mut b = FunctionBuilder::new("urem8", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c8 = b.iconst(8, Type::I64);
    let r = b.urem(x, c8);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    rax,0x7
        // CHECK-NEXT: mov    rcx,rdi
        // CHECK-NEXT: and    rcx,rax
        // CHECK-NEXT: mov    rax,rcx
        // CHECK-NOT: div
        // CHECK: ret
        ",
    );
}

/// Complex constant expression: all arithmetic folded away
#[test]
fn asm_constfold_complex() {
    let mut b = FunctionBuilder::new("cf_complex", &[], &[Type::I64]);
    let c5 = b.iconst(5, Type::I64);
    let c3 = b.iconst(3, Type::I64);
    let c10 = b.iconst(10, Type::I64);
    let c2 = b.iconst(2, Type::I64);
    let c1 = b.iconst(1, Type::I64);
    let a = b.mul(c5, c3);
    let d = b.sub(c10, c2);
    let sum = b.add(a, d);
    let r = b.udiv(sum, c1);
    b.ret(Some(r));

    // The e-graph fully folds (5*3)+(10-2)/1 = 23 = 0x17.
    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: mov    rax,0x17
        // CHECK-NOT: imul
        // CHECK-NOT: div
        // CHECK-NOT: add
        // CHECK-NOT: sub
        // CHECK-NEXT: ret
        ",
    );
}

/// Nested max: two sub+cmov pairs in sequence
#[test]
fn asm_nested_cmov() {
    let mut b = FunctionBuilder::new("max3", &[Type::I64, Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let cond1 = b.icmp(CondCode::Sgt, params[0], params[1]);
    let m1 = b.select(cond1, params[0], params[1]);
    let cond2 = b.icmp(CondCode::Sgt, m1, params[2]);
    let r = b.select(cond2, m1, params[2]);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // max3 = max(max(a, b), c) — two nested cmov/cmp pairs.
        // Each icmp lowers to a non-destructive CMP.
        // CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: cmovg  {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: cmovg  {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: ret
        ",
    );
}

/// Strength reduction + addressing mode combined: mul(x,3)+y
#[test]
fn asm_combined_sr_addr() {
    let mut b = FunctionBuilder::new("sr_addr", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let c3 = b.iconst(3, Type::I64);
    let m = b.mul(params[0], c3);
    let r = b.add(m, params[1]);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: lea    rcx,[rdi+rdi*2]
        // CHECK-NEXT: lea    rax,[rsi+rcx*1]
        // CHECK-NEXT: ret
        ",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// New algebraic rule tests: verify new egraph optimizations via IR/asm output
// ═══════════════════════════════════════════════════════════════════════════════

// ── Division/remainder identity rules ────────────────────────────────────────

/// sdiv(x, 1) should be eliminated — just returns x
#[test]
fn ir_sdiv_by_one() {
    let mut b = FunctionBuilder::new("sdiv1", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let one = b.iconst(1, Type::I64);
    let r = b.sdiv(x, one);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sdiv1
        // CHECK: v0 = param(0, I64)
        // CHECK-NOT: x86_idiv
        // CHECK: ret v0
        ",
    );
}

/// udiv(x, 1) should be eliminated — just returns x
#[test]
fn ir_udiv_by_one() {
    let mut b = FunctionBuilder::new("udiv1", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let one = b.iconst(1, Type::I64);
    let r = b.udiv(x, one);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function udiv1
        // CHECK: v0 = param(0, I64)
        // CHECK-NOT: x86_div
        // CHECK: ret v0
        ",
    );
}

/// srem(x, 1) should fold to constant 0
#[test]
fn ir_srem_by_one() {
    let mut b = FunctionBuilder::new("srem1", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let one = b.iconst(1, Type::I64);
    let r = b.srem(x, one);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function srem1
        // CHECK: iconst(0, I64)
        // CHECK-NOT: x86_idiv
        ",
    );
}

/// urem(x, 1) should fold to constant 0
#[test]
fn ir_urem_by_one() {
    let mut b = FunctionBuilder::new("urem1", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let one = b.iconst(1, Type::I64);
    let r = b.urem(x, one);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function urem1
        // CHECK: iconst(0, I64)
        // CHECK-NOT: x86_div
        ",
    );
}

/// sdiv(x, -1) should become negation (sub 0, x)
#[test]
fn ir_sdiv_by_neg_one() {
    let mut b = FunctionBuilder::new("sdiv_neg1", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let neg1 = b.iconst(-1, Type::I64);
    let r = b.sdiv(x, neg1);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sdiv_neg1
        // CHECK-NOT: x86_idiv
        ",
    );
}

// ── Select simplification ────────────────────────────────────────────────────

/// select(cond, a, a) should collapse to just a
#[test]
fn ir_select_same_arms() {
    let mut b = FunctionBuilder::new("sel_same", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let flags = b.icmp(CondCode::Slt, params[0], params[1]);
    let r = b.select(flags, params[0], params[0]);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sel_same
        // CHECK: v0 = param(0, I64)
        // CHECK-NOT: x86_cmov
        // CHECK: ret v0
        ",
    );
}

// ── Extension folding ────────────────────────────────────────────────────────

/// sext(i32->i64, sext(i8->i32, x)) should become sext(i8->i64, x)
#[test]
fn ir_sext_chain_folds() {
    let mut b = FunctionBuilder::new("sext_chain", &[Type::I8], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let s32 = b.sext(x, Type::I32);
    let s64 = b.sext(s32, Type::I64);
    b.ret(Some(s64));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sext_chain
        // CHECK: v0 = param(0, I8)
        // CHECK: x86_movsx(I8 -> I64)(v0)
        // CHECK-NOT: x86_movsx(I32 -> I64)
        ",
    );
}

/// zext(i32->i64, zext(i8->i32, x)) should become zext(i8->i64, x)
#[test]
fn ir_zext_chain_folds() {
    let mut b = FunctionBuilder::new("zext_chain", &[Type::I8], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let z32 = b.zext(x, Type::I32);
    let z64 = b.zext(z32, Type::I64);
    b.ret(Some(z64));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function zext_chain
        // CHECK: v0 = param(0, I8)
        // CHECK: x86_movzx(I8 -> I64)(v0)
        // CHECK-NOT: x86_movzx(I32 -> I64)
        ",
    );
}

// ── Bitwise complement ───────────────────────────────────────────────────────

/// x & ~x should fold to 0
#[test]
fn ir_and_complement_is_zero() {
    let mut b = FunctionBuilder::new("and_comp", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let neg1 = b.iconst(-1, Type::I64);
    let not_x = b.xor(x, neg1);
    let r = b.and(x, not_x);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function and_comp
        // CHECK: iconst(0, I64)
        // CHECK-NOT: x86_and
        ",
    );
}

/// x | ~x should fold to -1
#[test]
fn ir_or_complement_is_all_ones() {
    let mut b = FunctionBuilder::new("or_comp", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let neg1 = b.iconst(-1, Type::I64);
    let not_x = b.xor(x, neg1);
    let r = b.or(x, not_x);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function or_comp
        // CHECK: iconst(-1, I64)
        // CHECK-NOT: x86_or
        ",
    );
}

// ── Or annihilation ──────────────────────────────────────────────────────────

/// x | -1 should fold to -1
#[test]
fn ir_or_all_ones() {
    let mut b = FunctionBuilder::new("or_neg1", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let neg1 = b.iconst(-1, Type::I64);
    let r = b.or(x, neg1);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function or_neg1
        // CHECK: iconst(-1, I64)
        // CHECK-NOT: x86_or
        ",
    );
}

// ── Negation distribution ────────────────────────────────────────────────────

/// -(a + b) with constant a, b should fold completely
/// -(3 + 5) = -(8) = -8 via distribution + constant folding
#[test]
fn ir_neg_add_constants_fold() {
    let mut b = FunctionBuilder::new("neg_add", &[], &[Type::I64]);
    let c3 = b.iconst(3, Type::I64);
    let c5 = b.iconst(5, Type::I64);
    let sum = b.add(c3, c5);
    let r = b.neg(sum);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function neg_add
        // CHECK: iconst(-8, I64)
        // CHECK-NOT: x86_sub
        // CHECK-NOT: x86_add
        ",
    );
}

// ── Unified saturation: strength -> shift combining (cross-category) ─────────

/// Shl(Mul(x, 2), 3) should become Shl(x, 4) via unified saturation:
/// strength reduces Mul(x,2)->Shl(x,1), then shift combining merges Shl(Shl(x,1),3)->Shl(x,4).
/// This is impossible with phased execution (algebraic finishes before strength fires).
#[test]
fn ir_unified_strength_shift_combine() {
    let mut b = FunctionBuilder::new("unified_sc", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c2 = b.iconst(2, Type::I64);
    let c3 = b.iconst(3, Type::I64);
    let mul2 = b.mul(x, c2);
    let r = b.shl(mul2, c3);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function unified_sc
        // CHECK: v0 = param(0, I64)
        // CHECK: x86_shl_imm(4)(v0)
        ",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complex multi-rule chain tests
// ═══════════════════════════════════════════════════════════════════════════════

/// Realistic array access: base + (idx % 1) * 8
/// Chain: urem(idx, 1) -> 0 (identity), mul(0, 8) -> 0 (annihilation),
/// add(base, 0) -> base (identity). Entire index computation evaporates.
#[test]
fn ir_chain_rem1_kills_index() {
    let mut b = FunctionBuilder::new("rem1_idx", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let base = params[0];
    let idx = params[1];
    let one = b.iconst(1, Type::I64);
    let c8 = b.iconst(8, Type::I64);
    let rem = b.urem(idx, one); // -> 0
    let offset = b.mul(rem, c8); // -> mul(0, 8) -> 0
    let r = b.add(base, offset); // -> add(base, 0) -> base
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function rem1_idx
        // CHECK: v0 = param(0, I64)
        // CHECK-NOT: x86_div
        // CHECK-NOT: x86_imul
        // CHECK: ret v0
        ",
    );
}

/// Complement chain: (a & ~a) | (b & ~b) should fold to 0.
/// Each And(x, ~x) folds to 0, then Or(0, 0) folds to 0.
#[test]
fn ir_chain_double_complement_or() {
    let mut b = FunctionBuilder::new("dbl_comp", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let neg1 = b.iconst(-1, Type::I64);
    let not_a = b.xor(params[0], neg1);
    let lhs = b.and(params[0], not_a); // -> 0
    let not_b = b.xor(params[1], neg1);
    let rhs = b.and(params[1], not_b); // -> 0
    let r = b.or(lhs, rhs); // -> or(0, 0) -> 0
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function dbl_comp
        // CHECK: iconst(0, I64)
        // CHECK-NOT: x86_and
        // CHECK-NOT: x86_or
        // CHECK-NOT: x86_xor
        ",
    );
}

/// Strength + reassociation + constant folding chain:
/// (x * 4 + 3) + 5 should produce addr(x, 8) via:
///   - reassociation: (x*4 + 3) + 5 -> x*4 + (3+5) -> x*4 + 8
///   - strength: x*4 -> shl(x, 2)
///   - addr_mode: add(shl(x,2), 8) -> addr or lea with disp
#[test]
fn ir_chain_strength_reassoc_addr() {
    let mut b = FunctionBuilder::new("sr_reassoc", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c4 = b.iconst(4, Type::I64);
    let c3 = b.iconst(3, Type::I64);
    let c5 = b.iconst(5, Type::I64);
    let mul4 = b.mul(x, c4);
    let inner = b.add(mul4, c3);
    let r = b.add(inner, c5);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sr_reassoc
        // CHECK: param(0, I64)
        // CHECK-NOT: x86_imul
        // CHECK: addr(scale=4
        ",
    );
}

/// Verify strength+shift combining in asm: Shl(Mul(x, 2), 3) -> single shl by 4.
/// The asm should have a single shl instruction with immediate 4, no imul.
#[test]
fn asm_unified_strength_shift_combine() {
    let mut b = FunctionBuilder::new("unified_asm", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let c2 = b.iconst(2, Type::I64);
    let c3 = b.iconst(3, Type::I64);
    let mul2 = b.mul(x, c2);
    let r = b.shl(mul2, c3);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK-NOT: imul
        // CHECK: shl    {{[a-z0-9]+}},0x4
        // CHECK: ret
        ",
    );
}

/// Division identity in asm: sdiv(x, 1) should produce no idiv instruction.
/// The function should just move the param to rax and return.
#[test]
fn asm_sdiv_by_one_no_idiv() {
    let mut b = FunctionBuilder::new("sdiv1_asm", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let one = b.iconst(1, Type::I64);
    let r = b.sdiv(x, one);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK-NOT: idiv
        // CHECK-NOT: cqo
        // CHECK: ret
        ",
    );
}

/// Complement in asm: x & ~x should produce xor (zeroing), no and instruction.
#[test]
fn asm_and_complement_to_xor_zero() {
    let mut b = FunctionBuilder::new("and_comp_asm", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let neg1 = b.iconst(-1, Type::I64);
    let not_x = b.xor(x, neg1);
    let r = b.and(x, not_x);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: xor    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK-NOT: and
        // CHECK: ret
        ",
    );
}

/// Multi-op constant folding chain:
/// ((10 / 1) % 1) | -1 should fold entirely to -1.
/// div identity -> rem identity -> or annihilation, all constant.
#[test]
fn ir_chain_div_rem_or_fold() {
    let mut b = FunctionBuilder::new("fold_chain", &[], &[Type::I64]);
    let c10 = b.iconst(10, Type::I64);
    let c1 = b.iconst(1, Type::I64);
    let neg1 = b.iconst(-1, Type::I64);
    let d = b.sdiv(c10, c1); // -> 10
    let r = b.srem(d, c1); // -> 0
    let result = b.or(r, neg1); // -> -1
    b.ret(Some(result));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function fold_chain
        // CHECK: iconst(-1, I64)
        // CHECK-NOT: x86_idiv
        // CHECK-NOT: x86_or
        ",
    );
}

/// Extension folding + strength reduction chain:
/// zext(i8->i32, zext(i8->i8... wait, need valid chain))
/// sext(i8->i64, x) then mul by 4 -> sext + shl(2)
/// Tests that extension folding composes with downstream strength reduction.
#[test]
fn ir_chain_zext_fold_then_shift() {
    let mut b = FunctionBuilder::new("zext_shift", &[Type::I8], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let z32 = b.zext(x, Type::I32);
    let z64 = b.zext(z32, Type::I64);
    // Now multiply the extended value by 8 -> should become shl by 3
    let c8 = b.iconst(8, Type::I64);
    let r = b.mul(z64, c8);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function zext_shift
        // CHECK: v0 = param(0, I8)
        // CHECK: x86_movzx(I8 -> I64)(v0)
        // CHECK-NOT: x86_movzx(I32 -> I64)
        // CHECK-NOT: x86_imul
        // CHECK: x86_shl_imm(3)
        ",
    );
}

/// Realistic "clear then set" pattern: (x & 0) | y should fold to y.
/// annihilation: x & 0 -> 0, then identity: 0 | y -> y.
#[test]
fn ir_chain_mask_clear_then_or() {
    let mut b = FunctionBuilder::new("clear_set", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let zero = b.iconst(0, Type::I64);
    let cleared = b.and(params[0], zero); // -> 0
    let r = b.or(cleared, params[1]); // -> or(0, y) -> y
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function clear_set
        // CHECK-NOT: x86_and
        // CHECK-NOT: x86_or
        // CHECK: param(1, I64)
        // CHECK: ret
        ",
    );
}

/// Negation of negation via double Sub(0, _):
/// Sub(0, Sub(0, x)) should fold to x (double negation rule).
#[test]
fn ir_double_negation() {
    let mut b = FunctionBuilder::new("dbl_neg", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let neg1 = b.neg(x);
    let r = b.neg(neg1);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function dbl_neg
        // CHECK: v0 = param(0, I64)
        // CHECK-NOT: x86_sub
        // CHECK: ret v0
        ",
    );
}

/// Four-way chain: identity + annihilation + complement + strength.
/// ((x + 0) * (y & ~y)) + (z * 8)
///   x+0 -> x (identity), y & ~y -> 0 (complement), x * 0 -> 0 (annihilation),
///   0 + z*8 -> z*8 (identity), z*8 -> shl(z, 3) (strength)
/// Result: just shl(z, 3)
#[test]
fn ir_four_rule_chain() {
    let mut b = FunctionBuilder::new(
        "four_chain",
        &[Type::I64, Type::I64, Type::I64],
        &[Type::I64],
    );
    let params = b.params().to_vec();
    let x = params[0];
    let y = params[1];
    let z = params[2];
    let zero = b.iconst(0, Type::I64);
    let c8 = b.iconst(8, Type::I64);
    let neg1 = b.iconst(-1, Type::I64);

    let x_plus_0 = b.add(x, zero); // -> x
    let not_y = b.xor(y, neg1);
    let y_and_not_y = b.and(y, not_y); // -> 0
    let lhs = b.mul(x_plus_0, y_and_not_y); // -> x * 0 -> 0
    let rhs = b.mul(z, c8); // -> shl(z, 3)
    let r = b.add(lhs, rhs); // -> 0 + shl(z,3) -> shl(z,3)
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function four_chain
        // CHECK-NOT: x86_imul
        // CHECK-NOT: x86_and
        // CHECK-NOT: x86_xor
        // CHECK: addr(scale=8
        ",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-rule combination tests: chains of 3+ rules cooperating
// ═══════════════════════════════════════════════════════════════════════════════

/// sdiv(neg(x), -1) -> x via 2-rule chain:
/// sdiv_neg1: sdiv(sub(0,x), -1) = sub(0, sub(0,x))
/// double_neg: sub(0, sub(0, x)) = x
#[test]
fn ir_combo_sdiv_neg1_double_neg() {
    let mut b = FunctionBuilder::new("sdiv_dblneg", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let neg_x = b.neg(x);
    let neg1 = b.iconst(-1, Type::I64);
    let r = b.sdiv(neg_x, neg1);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sdiv_dblneg
        // CHECK: v0 = param(0, I64)
        // CHECK-NOT: x86_idiv
        // CHECK-NOT: x86_sub
        // CHECK: ret v0
        ",
    );
}

/// 5-rule chain across 3 categories: urem(y,1)->0, mul(0,4)->0, add(x,0)->x,
/// mul(x,2)->shl(x,1) [strength], shl(shl(x,1),3)->shl(x,4) [shift combine].
/// The entire computation collapses to a single shift.
#[test]
fn ir_combo_5rule_rem_annihil_strength_shift() {
    let mut b = FunctionBuilder::new("big5", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let (x, y) = (params[0], params[1]);
    let c1 = b.iconst(1, Type::I64);
    let c2 = b.iconst(2, Type::I64);
    let c3 = b.iconst(3, Type::I64);
    let c4 = b.iconst(4, Type::I64);
    let rem = b.urem(y, c1); // -> 0
    let scaled = b.mul(rem, c4); // -> 0*4 -> 0
    let base = b.add(x, scaled); // -> x+0 -> x
    let doubled = b.mul(base, c2); // -> x*2 -> shl(x,1) [strength]
    let r = b.shl(doubled, c3); // -> shl(shl(x,1),3) -> shl(x,4) [unified]
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function big5
        // CHECK: param(0, I64)
        // CHECK-NOT: x86_div
        // CHECK-NOT: x86_imul
        // CHECK: x86_shl_imm(4)
        ",
    );
}

/// complement + identity chain: (a & ~a) + b -> 0 + b -> b.
/// complement: and(a, xor(a,-1)) = 0
/// identity: add(0, b) = b
/// Two params in, one is dead (a), output is just b.
#[test]
fn ir_combo_complement_identity() {
    let mut b = FunctionBuilder::new("comp_id", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let neg1 = b.iconst(-1, Type::I64);
    let not_a = b.xor(params[0], neg1);
    let dead = b.and(params[0], not_a); // -> 0
    let r = b.add(dead, params[1]); // -> 0 + b -> b
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function comp_id
        // CHECK-NOT: x86_and
        // CHECK-NOT: x86_xor
        // CHECK-NOT: x86_add
        // CHECK: param(1, I64)
        // CHECK: ret
        ",
    );
}

/// Extension fold + complement: zext(zext(x)) & ~(zext(zext(x))) -> 0.
/// ext_fold: zext(i32, zext(i8, x)) = zext(i64, x)
/// complement: val & ~val = 0
#[test]
fn ir_combo_ext_fold_complement() {
    let mut b = FunctionBuilder::new("ext_comp", &[Type::I8], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let z32 = b.zext(x, Type::I32);
    let z64 = b.zext(z32, Type::I64);
    let neg1 = b.iconst(-1, Type::I64);
    let not_z = b.xor(z64, neg1);
    let r = b.and(z64, not_z);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function ext_comp
        // CHECK: iconst(0, I64)
        // CHECK-NOT: x86_and
        // CHECK-NOT: x86_movzx
        ",
    );
}

/// 4-rule chain: or_annihil + div_id + rem_id + constant_fold.
/// (x | -1) / 1 + (y % 1) -> -1 / 1 + 0 -> -1 + 0 -> -1.
/// Both params become dead; output is a single constant.
#[test]
fn ir_combo_annihil_div_rem_fold() {
    let mut b = FunctionBuilder::new("ann_div_rem", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let neg1 = b.iconst(-1, Type::I64);
    let c1 = b.iconst(1, Type::I64);
    let or_neg1 = b.or(params[0], neg1); // -> -1
    let div1 = b.sdiv(or_neg1, c1); // -> -1 / 1 -> -1
    let rem1 = b.urem(params[1], c1); // -> 0
    let r = b.add(div1, rem1); // -> -1 + 0 -> -1
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function ann_div_rem
        // CHECK: iconst(-1, I64)
        // CHECK-NOT: x86_or
        // CHECK-NOT: x86_idiv
        // CHECK-NOT: x86_div
        // CHECK-NOT: x86_add
        ",
    );
}

/// select_same + double_neg: select(c, -(-x), -(-x)) -> x.
/// double_neg: sub(0, sub(0, x)) = x
/// select_same: select(c, a, a) = a
#[test]
fn ir_combo_select_double_neg() {
    let mut b = FunctionBuilder::new("sel_dblneg", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let zero = b.iconst(0, Type::I64);
    let neg_x = b.neg(params[0]);
    let dbl_neg = b.neg(neg_x);
    let cond = b.icmp(CondCode::Slt, params[1], zero);
    let r = b.select(cond, dbl_neg, dbl_neg);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function sel_dblneg
        // CHECK: v0 = param(0, I64)
        // CHECK-NOT: x86_sub
        // CHECK-NOT: x86_cmov
        // CHECK: ret v0
        ",
    );
}

/// double_neg + reassociation + addr_mode: -(-(p + 3)) + 5.
/// double_neg removes the two negations -> p + 3.
/// Then (p + 3) + 5: reassociation + constant fold -> p + 8.
/// Addr mode captures the add with constant.
#[test]
fn ir_combo_dblneg_reassoc_addr() {
    let mut b = FunctionBuilder::new("dn_reassoc", &[Type::I64], &[Type::I64]);
    let p = b.params().to_vec()[0];
    let c3 = b.iconst(3, Type::I64);
    let c5 = b.iconst(5, Type::I64);
    let sum = b.add(p, c3);
    let neg1 = b.neg(sum);
    let neg2 = b.neg(neg1);
    let r = b.add(neg2, c5);
    b.ret(Some(r));

    check_ir(
        b.finalize().unwrap(),
        "
        // CHECK-LABEL: function dn_reassoc
        // CHECK: param(0, I64)
        // CHECK-NOT: x86_sub
        // CHECK-NOT: x86_imul
        ",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// cmp-vs-sub isel: X86Sub with a dead difference becomes a non-destructive
// CmpRR at lowering time. When the difference is live (e.g. a Sub expression
// shares the X86Sub class), the destructive SUB is preserved.
// ═══════════════════════════════════════════════════════════════════════════════

/// Icmp against a small immediate: isel creates X86CmpI and extraction
/// picks it over Proj1(X86Sub). Lowers to `cmp r, imm` (no register-to-hold
/// the immediate, no destructive sub). LHS-iconst case left for future work.
#[test]
fn asm_icmp_with_imm_uses_cmp_ri() {
    let mut b = FunctionBuilder::new("icmp_imm", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let one = b.iconst(1, Type::I64);
    let cond = b.icmp(CondCode::Sgt, x, one);
    let yes = b.iconst(42, Type::I64);
    let no = b.iconst(0, Type::I64);
    let r = b.select(cond, yes, no);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // The compare fuses the `1` as an immediate; no separate mov for it.
        // CHECK-NOT: sub
        // CHECK: cmp    {{[a-z0-9]+}},0x1
        // CHECK: cmov
        // CHECK: ret
        ",
    );
}

/// Icmp against zero: isel creates X86CmpI(0) which lowers to `test r, r`
/// (2 bytes, same flags as `cmp r, 0`).
#[test]
fn asm_icmp_zero_uses_test() {
    let mut b = FunctionBuilder::new("icmp_zero", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let zero = b.iconst(0, Type::I64);
    let cond = b.icmp(CondCode::Ne, x, zero);
    let one = b.iconst(1, Type::I64);
    let r = b.select(cond, one, zero);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK-NOT: sub
        // CHECK-NOT: cmp
        // CHECK: test   {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: cmov
        // CHECK: ret
        ",
    );
}

/// Icmp alone with no separate Sub on the operands: difference is dead, so we
/// emit `cmp` (not `mov + sub`).
#[test]
fn asm_icmp_alone_uses_cmp() {
    let mut b = FunctionBuilder::new("icmp_only", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let cond = b.icmp(CondCode::Sgt, params[0], params[1]);
    let one = b.iconst(1, Type::I64);
    let zero = b.iconst(0, Type::I64);
    let r = b.select(cond, one, zero);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK: cmov
        // CHECK: ret
        ",
    );
}

/// When the same (a, b) is consumed both as `Sub(a, b)` (difference live) and
/// as `Icmp(Sgt, a, b)` (flags), the shared X86Sub stays destructive so the
/// flags come from the same instruction. Must see a `sub` (not a `cmp`) for
/// the shared op. The e-graph isel hashcons merges the two X86Sub nodes.
#[test]
fn asm_shared_sub_keeps_destructive() {
    let mut b = FunctionBuilder::new("shared_sub", &[Type::I64, Type::I64], &[Type::I64]);
    let params = b.params().to_vec();
    let diff = b.sub(params[0], params[1]);
    let cond = b.icmp(CondCode::Sgt, params[0], params[1]);
    let zero = b.iconst(0, Type::I64);
    let r = b.select(cond, diff, zero);
    b.ret(Some(r));

    check_asm(
        b.finalize().unwrap(),
        "
        // The shared X86Sub(a, b) feeds both Proj0 (the difference) and
        // Proj1 (the flags). Keep the destructive SUB; no CMP needed.
        // CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
        // CHECK-NOT: cmp
        // CHECK: ret
        ",
    );
}
