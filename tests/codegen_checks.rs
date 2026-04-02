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
        // CHECK: branch v3 block1() block2()
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
        // CHECK: jump block1(v0, v0)
        // CHECK-LABEL: block1(p0: I64, p1: I64):
        // CHECK: v2 = block_param(b1, 1, I64)
        // CHECK: v4 = iconst(1, I64)
        // CHECK-NEXT: v5 = addr(scale=1, disp=0)(v2, v4)
        // CHECK: v6 = param(0, I64)
        // CHECK-NEXT: v7 = x86_sub(v5, v6)
        // CHECK-NEXT: v8 = proj1(v7)
        // CHECK: v1 = block_param(b1, 0, I64)
        // CHECK-NEXT: v3 = addr(scale=1, disp=0)(v1, v2)
        // CHECK: branch v8 block1(v3, v5) block2(v3)
        // CHECK-LABEL: block2(p0: I64):
        // CHECK: ret v9
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
        // CHECK: xor    rdx,rdx
        // CHECK: sub    rax,rdx
        // CHECK-NEXT: mov    rax,rdx
        // CHECK-NEXT: cmovg  rax,r8
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
        // CHECK: sub    rsp,0x8
        // CHECK-NEXT: mov    rdi,0x1
        // CHECK-NEXT: mov    rsi,0x2
        // CHECK-NEXT: mov    rdx,0x3
        // CHECK-NEXT: mov    rcx,0x4
        // CHECK-NEXT: call
        // CHECK-NEXT: add    rsp,0x8
        // CHECK-NEXT: ret
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
        // CHECK: mov    rcx,rdi
        // CHECK-NEXT: sub    rcx,rsi
        // CHECK: jg
        // CHECK: mov    rdx,0x1
        // CHECK-NEXT: lea    rcx,[rax+rdx*1]
        // CHECK-NEXT: mov    rax,rcx
        // CHECK-NEXT: jmp
        // CHECK: mov    rdx,0x1
        // CHECK-NEXT: lea    rcx,[rax+rdx*1]
        // CHECK-NEXT: mov    rax,rcx
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
        // CHECK: xor    rax,rax
        // CHECK-NEXT: mov    r8,rax
        // CHECK-NEXT: mov    rsi,rax
        // CHECK: mov    rax,0x1
        // CHECK-NEXT: lea    rdx,[rsi+rax*1]
        // CHECK-NEXT: mov    rcx,rdx
        // CHECK-NEXT: sub    rcx,rdi
        // CHECK-NEXT: lea    rax,[r8+rsi*1]
        // CHECK-NEXT: jge
        // CHECK-NEXT: mov    r8,rax
        // CHECK-NEXT: mov    rsi,rdx
        // CHECK-NEXT: jmp
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
        // CHECK: mov    rcx,rdi
        // CHECK-NEXT: sub    rcx,rsi
        // CHECK-NEXT: mov    r8,rsi
        // CHECK-NEXT: cmovg  r8,rdi
        // CHECK-NEXT: mov    rax,r8
        // CHECK-NEXT: sub    rax,rdx
        // CHECK-NEXT: mov    rax,rdx
        // CHECK-NEXT: cmovg  rax,r8
        // CHECK-NEXT: ret
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
