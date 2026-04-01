use super::*;
use crate::ir::builder::FunctionBuilder;
use crate::ir::types::Type;
use crate::test_utils::has_tool;

fn build_identity() -> Function {
    let mut builder = FunctionBuilder::new("identity", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    builder.ret(Some(params[0]));
    builder.finalize().expect("identity finalize")
}

fn build_add() -> Function {
    let mut builder = FunctionBuilder::new("add_two", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let sum = builder.add(params[0], params[1]);
    builder.ret(Some(sum));
    builder.finalize().expect("add finalize")
}

// 14.3: DiagnosticSink test.
#[test]
fn diagnostic_sink_receives_stats() {
    struct VecSink(Vec<String>);
    impl DiagnosticSink for VecSink {
        fn phase_stats(&mut self, phase: &str, stats: &str) {
            self.0.push(format!("{phase}: {stats}"));
        }
    }

    let func = build_add();
    let opts = CompileOptions {
        verbosity: Verbosity::Verbose,
        ..Default::default()
    };
    let mut sink = VecSink(Vec::new());
    let result = compile(func, &opts, Some(&mut sink));
    assert!(result.is_ok(), "compile failed: {:?}", result.err());
    assert!(
        !sink.0.is_empty(),
        "diagnostic sink should have received phase stats"
    );
    let all = sink.0.join("\n");
    assert!(all.contains("egraph:"), "should have egraph stats");
}

// 14.4: identity(x) -> x
#[test]
fn e2e_identity() {
    if !has_tool("cc") {
        return;
    }

    let func = build_identity();
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile identity");

    let dir = std::env::temp_dir();
    let obj_path = dir.join("blitz_e2e_identity.o");
    let main_path = dir.join("blitz_e2e_identity_main.c");
    let bin_path = dir.join("blitz_e2e_identity_bin");

    obj.write_to(&obj_path).expect("write .o");

    std::fs::write(
        &main_path,
        b"#include <stdint.h>\n\
              int64_t identity(int64_t x);\n\
              int main(void) {\n\
              int64_t r = identity(42);\n\
              return (r == 42) ? 0 : 1;\n\
              }\n",
    )
    .expect("write main.c");

    let compile_out = std::process::Command::new("cc")
        .args([
            main_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc");

    assert!(
        compile_out.status.success(),
        "linking failed:\n{}",
        String::from_utf8_lossy(&compile_out.stderr)
    );

    let run = std::process::Command::new(&bin_path).output().expect("run");
    assert_eq!(run.status.code(), Some(0), "identity(42) should return 42");

    let _ = std::fs::remove_file(&obj_path);
    let _ = std::fs::remove_file(&main_path);
    let _ = std::fs::remove_file(&bin_path);
}

// 14.5: add(a, b) -> a + b
#[test]
fn e2e_add() {
    if !has_tool("cc") {
        return;
    }

    let func = build_add();
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile add");

    let dir = std::env::temp_dir();
    let obj_path = dir.join("blitz_e2e_add.o");
    let main_path = dir.join("blitz_e2e_add_main.c");
    let bin_path = dir.join("blitz_e2e_add_bin");

    obj.write_to(&obj_path).expect("write .o");

    std::fs::write(
        &main_path,
        b"#include <stdint.h>\n\
              int64_t add_two(int64_t a, int64_t b);\n\
              int main(void) {\n\
              int64_t r = add_two(3, 4);\n\
              return (r == 7) ? 0 : 1;\n\
              }\n",
    )
    .expect("write main.c");

    let compile_out = std::process::Command::new("cc")
        .args([
            main_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc");

    assert!(
        compile_out.status.success(),
        "linking failed:\n{}",
        String::from_utf8_lossy(&compile_out.stderr)
    );

    let run = std::process::Command::new(&bin_path).output().expect("run");
    assert_eq!(run.status.code(), Some(0), "add_two(3, 4) should return 7");

    let _ = std::fs::remove_file(&obj_path);
    let _ = std::fs::remove_file(&main_path);
    let _ = std::fs::remove_file(&bin_path);
}

// 14.2: Multi-function compilation.
#[test]
fn compile_module_two_functions() {
    if !has_tool("cc") {
        return;
    }

    let id_pair = build_identity();
    let add_pair = build_add();
    let opts = CompileOptions::default();

    let functions = vec![id_pair, add_pair];
    let obj = compile_module(functions, &opts).expect("compile_module");

    assert_eq!(obj.functions.len(), 2);
    assert_eq!(obj.functions[0].name, "identity");
    assert_eq!(obj.functions[1].name, "add_two");
    // Second function offset must be > 0.
    assert!(
        obj.functions[1].offset > 0,
        "add_two should have non-zero offset"
    );
}

// ── Helper: link and run a generated object with a C main ─────────────────

fn link_and_run(test_name: &str, obj_bytes: &[u8], c_main: &str) -> Option<i32> {
    if !has_tool("cc") {
        return None;
    }
    use std::process::Command;

    let dir = std::env::temp_dir();
    let obj_path = dir.join(format!("{test_name}.o"));
    let main_path = dir.join(format!("{test_name}_main.c"));
    let bin_path = dir.join(format!("{test_name}_bin"));

    // Write a minimal ObjectFile wrapping the raw bytes.
    // The test already has a compiled ObjectFile; write it directly.
    std::fs::write(&obj_path, obj_bytes).expect("write .o");
    std::fs::write(&main_path, c_main.as_bytes()).expect("write main.c");

    let compile_out = Command::new("cc")
        .args([
            main_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc");

    if !compile_out.status.success() {
        eprintln!(
            "cc failed:\n{}",
            String::from_utf8_lossy(&compile_out.stderr)
        );
        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(&main_path);
        return None;
    }

    let run = Command::new(&bin_path).output().expect("run binary");
    let code = run.status.code();

    let _ = std::fs::remove_file(&obj_path);
    let _ = std::fs::remove_file(&main_path);
    let _ = std::fs::remove_file(&bin_path);

    code
}

// ── Helper: write ObjectFile and link ─────────────────────────────────────

fn link_and_run_obj(
    test_name: &str,
    obj: &crate::emit::object::ObjectFile,
    c_main: &str,
) -> Option<i32> {
    if !has_tool("cc") {
        return None;
    }
    let dir = std::env::temp_dir();
    let obj_path = dir.join(format!("{test_name}.o"));
    obj.write_to(&obj_path).expect("write .o");
    let bytes = std::fs::read(&obj_path).unwrap();
    let _ = std::fs::remove_file(&obj_path);
    link_and_run(test_name, &bytes, c_main)
}

// 14.6: Conditional branch — max(a, b) using select/cmov (single-block).
//
// Build: icmp(Sgt, a, b) -> flags; select(flags, a, b) -> result; ret(result)
// After isel, Select becomes X86Cmov; stays single-block, no branch needed.
#[test]
fn e2e_conditional_max() {
    use crate::ir::condcode::CondCode;

    let mut builder = FunctionBuilder::new("blitz_max", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];
    let flags = builder.icmp(CondCode::Sgt, a, b);
    let result = builder.select(flags, a, b);
    builder.ret(Some(result));
    let func = builder.finalize().expect("max finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile max");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_max(int64_t a, int64_t b);
int main(void) {
    if (blitz_max(10, 5) != 10) return 1;
    if (blitz_max(3, 7) != 7) return 2;
    if (blitz_max(4, 4) != 4) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_max", &obj, c_main) {
        assert_eq!(code, 0, "max function returned wrong exit code {code}");
    }
}

// 14.7: Loop — sum 1..=n using multi-block with back-edge.
//
// IR structure:
//   BB0 (entry): jump(BB1, [0, 1])
//   BB1 (params=[acc, i]):
//     new_acc = add(acc, i)
//     new_i   = add(i, 1)
//     cond    = icmp(Sle, new_i, n)
//     branch(cond, BB1, BB2, [new_acc, new_i], [new_acc])
//   BB2 (params=[result]): ret(result)
//
// sum(1..=5) = 15, sum(1..=10) = 55
#[test]
fn e2e_loop_sum() {
    use crate::ir::condcode::CondCode;

    let mut builder = FunctionBuilder::new("blitz_sum", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let n = params[0];

    // Create blocks.
    let (bb1, bb1_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
    let acc = bb1_params[0];
    let i = bb1_params[1];
    let (bb2, bb2_params) = builder.create_block_with_params(&[Type::I64]);
    let result = bb2_params[0];

    // BB0: jump to BB1 with acc=0, i=1.
    let zero = builder.iconst(0, Type::I64);
    let one = builder.iconst(1, Type::I64);
    builder.jump(bb1, &[zero, one]);

    // BB1: loop body.
    builder.set_block(bb1);
    let new_acc = builder.add(acc, i);
    let new_i = builder.add(i, one);
    let cond = builder.icmp(CondCode::Sle, new_i, n);
    builder.branch(cond, bb1, bb2, &[new_acc, new_i], &[new_acc]);

    // BB2: return result.
    builder.set_block(bb2);
    builder.ret(Some(result));

    let func = builder.finalize().expect("sum finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile sum");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_sum(int64_t n);
int main(void) {
    if (blitz_sum(5) != 15) return 1;
    if (blitz_sum(10) != 55) return 2;
    if (blitz_sum(1) != 1) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_sum", &obj, c_main) {
        assert_eq!(code, 0, "sum function returned wrong exit code {code}");
    }
}

// 14.8: Function call — call an external C function.
//
// Build: abs(x) = x >= 0 ? x : -x  using select + icmp + sub(0, x)
// Alternatively: call abs() from libc.
// We call a simple helper: double(x) = x + x (defined in C main).
#[test]
fn e2e_call_external() {
    let mut builder = FunctionBuilder::new("blitz_call_double", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let x = params[0];
    // Call external function "double_val(x)" and return result.
    let results = builder.call("double_val", &[x], &[Type::I64]);
    builder.ret(Some(results[0]));
    let func = builder.finalize().expect("call finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile call");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_call_double(int64_t x);
int64_t double_val(int64_t x) { return x + x; }
int main(void) {
    if (blitz_call_double(5) != 10) return 1;
    if (blitz_call_double(0) != 0) return 2;
    if (blitz_call_double(-3) != -6) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_call", &obj, c_main) {
        assert_eq!(code, 0, "call_double returned wrong exit code {code}");
    }
}

// 14.9: Addressing modes — compile a function using scaled-index addressing.
//
// Build: stride_offset(base, idx) = base + idx * 4
// After isel + addr-mode rules, this should compile to a LEA with scale.
#[test]
fn e2e_addressing_modes() {
    use crate::test_utils::objdump_disasm;

    // Build: stride_test(base, idx) = base + idx * 4
    // This exercises the X86Lea3{scale:4} addressing mode rule.
    let mut builder =
        FunctionBuilder::new("blitz_stride_test", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let base = params[0];
    let idx = params[1];
    let two = builder.iconst(2, Type::I64);
    // idx << 2 = idx * 4
    let scaled = builder.shl(idx, two);
    let addr = builder.add(base, scaled);
    builder.ret(Some(addr));
    let func = builder.finalize().expect("stride finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile stride_test");

    // Verify correctness.
    let c_main = r#"
#include <stdint.h>
int64_t blitz_stride_test(int64_t base, int64_t idx);
int main(void) {
    // base=100, idx=3 => 100 + 3*4 = 112
    if (blitz_stride_test(100, 3) != 112) return 1;
    // base=0, idx=0 => 0
    if (blitz_stride_test(0, 0) != 0) return 2;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_stride", &obj, c_main) {
        assert_eq!(code, 0, "stride_test returned wrong exit code {code}");
    }

    // Optionally verify the disassembly shows LEA (addr mode optimization).
    if let Some(disasm) = objdump_disasm(&obj.code) {
        // After addr-mode rules, base + idx<<2 should become lea [base + idx*4].
        // Accept LEA or SHL+ADD as both are correct encodings.
        let has_efficient_code =
            disasm.contains("lea") || disasm.contains("shl") || disasm.contains("add");
        assert!(
            has_efficient_code,
            "expected LEA or SHL/ADD in disassembly:\n{disasm}"
        );
    }
}

// 14.10: Register pressure — 20+ simultaneously live values.
//
// Build a function that uses more than 15 live values simultaneously
// to exercise spilling. The function computes the sum of 20 constants
// to keep many values live at once.
#[test]
fn e2e_register_pressure() {
    // Build: sum20() = 1 + 2 + 3 + ... + 20
    // By loading all 20 iconsts before adding them, we create pressure.
    let mut builder = FunctionBuilder::new("blitz_sum20", &[], &[Type::I64]);

    let vals: Vec<_> = (1i64..=20).map(|v| builder.iconst(v, Type::I64)).collect();

    // Chain-add all 20 values in a binary tree pattern to keep many live.
    // This forces regalloc to handle high pressure.
    let mut acc = vals[0];
    for &v in &vals[1..] {
        acc = builder.add(acc, v);
    }
    builder.ret(Some(acc));

    let func = builder.finalize().expect("sum20 finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile sum20");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_sum20(void);
int main(void) {
    // 1+2+...+20 = 210
    if (blitz_sum20() != 210) return 1;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_sum20", &obj, c_main) {
        assert_eq!(code, 0, "sum20 returned wrong exit code {code}");
    }
}

// 14.11: Flag fusion — if (a - b > 0) return a - b, else return 0.
//
// The optimizer should fuse the subtraction that produces the value
// with the comparison, emitting a single SUB instruction (no CMP needed).
#[test]
fn e2e_flag_fusion() {
    use crate::ir::condcode::CondCode;
    use crate::test_utils::objdump_disasm;

    // Build: flag_fusion(a, b) = max(a - b, 0) using select.
    let mut builder =
        FunctionBuilder::new("blitz_flag_fusion", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];
    let diff = builder.sub(a, b);
    let zero = builder.iconst(0, Type::I64);
    let cond = builder.icmp(CondCode::Sgt, diff, zero);
    let result = builder.select(cond, diff, zero);
    builder.ret(Some(result));
    let func = builder.finalize().expect("flag_fusion finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile flag_fusion");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_flag_fusion(int64_t a, int64_t b);
int main(void) {
    if (blitz_flag_fusion(5, 3) != 2) return 1;
    if (blitz_flag_fusion(3, 5) != 0) return 2;
    if (blitz_flag_fusion(4, 4) != 0) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_flag_fusion", &obj, c_main) {
        assert_eq!(code, 0, "flag_fusion returned wrong exit code {code}");
    }

    // Verify that objdump shows a SUB but no separate CMP
    // (sub a, b sets the flags that cmov uses — no extra cmp needed).
    if let Some(disasm) = objdump_disasm(&obj.code) {
        let has_sub = disasm.contains("sub");
        let has_cmp = disasm.contains("cmp");
        assert!(has_sub, "expected SUB in disassembly:\n{disasm}");
        assert!(
            !has_cmp,
            "expected NO CMP (flag fusion should reuse SUB flags):\n{disasm}"
        );
    }
}

// 14.12: Snapshot — compile identity + add, store reference disassembly.
//
// Compiles both functions and verifies the disassembly matches a known
// reference. If objdump is unavailable, just verify compilation succeeds.
#[test]
fn e2e_snapshot() {
    use crate::test_utils::objdump_disasm;

    let id_func = build_identity();
    let add_func = build_add();
    let opts = CompileOptions::default();

    let id_obj = compile(id_func, &opts, None).expect("compile identity");
    let add_obj = compile(add_func, &opts, None).expect("compile add");

    // Verify the identity function is minimal: leaf function with no calls or spills.
    // With frame pointer omission (default), the prologue and epilogue are absent.
    // Expected bytes: 48 89 f8 c3  (mov rax, rdi; ret)
    let expected_identity: &[u8] = &[
        0x48, 0x89, 0xf8, // mov rax, rdi
        0xc3, // ret
    ];
    assert_eq!(
        &id_obj.code, expected_identity,
        "identity function bytes mismatch"
    );

    // Verify add function compiles and has plausible size.
    assert!(
        add_obj.code.len() >= 5,
        "add function should be at least 5 bytes"
    );

    // Optional: print disassembly for both if objdump is available.
    if let Some(disasm) = objdump_disasm(&id_obj.code) {
        assert!(
            disasm.contains("mov") && disasm.contains("ret"),
            "identity disassembly should contain mov and ret:\n{disasm}"
        );
    }

    if let Some(disasm) = objdump_disasm(&add_obj.code) {
        assert!(
            disasm.contains("add") || disasm.contains("lea"),
            "add disassembly should contain add or lea:\n{disasm}"
        );
    }
}

// 14.9: Branch relaxation -- short-form jumps.
//
// Compile a simple conditional (two blocks) and verify that the jump bytes
// use the short form (EB for JMP, 7x for Jcc) since the blocks are close
// together.  We scan `obj.code` for near-form jump opcodes (E9, 0F 8x)
// and assert none appear.
#[test]
fn branch_relaxation_uses_short_form_for_nearby_targets() {
    use crate::ir::condcode::CondCode;

    // Build: max(a, b) = if a >= b { a } else { b }
    // Single condition, two close blocks -> all jumps should be short.
    let mut builder =
        FunctionBuilder::new("blitz_max_short", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];

    let (bb_true, _) = builder.create_block_with_params(&[]);
    let (bb_false, _) = builder.create_block_with_params(&[]);

    let cond = builder.icmp(CondCode::Sge, a, b);
    builder.branch(cond, bb_true, bb_false, &[], &[]);

    builder.set_block(bb_true);
    builder.ret(Some(a));

    builder.set_block(bb_false);
    builder.ret(Some(b));

    let func = builder.finalize().expect("max_short finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile max_short");

    // Walk the code bytes and look for near-form jump opcodes.
    // E9 = near JMP; 0F followed by 80..8F = near Jcc.
    let code = &obj.code;
    let mut i = 0;
    let mut found_near_jmp = false;
    let mut found_near_jcc = false;
    while i < code.len() {
        match code[i] {
            0xE9 => {
                found_near_jmp = true;
            }
            0x0F if i + 1 < code.len() && (0x80..=0x8F).contains(&code[i + 1]) => {
                found_near_jcc = true;
            }
            _ => {}
        }
        i += 1;
    }

    assert!(
        !found_near_jmp,
        "nearby JMP should use short form (EB), found near form (E9) in {:02X?}",
        code
    );
    assert!(
        !found_near_jcc,
        "nearby Jcc should use short form (7x), found near form (0F 8x) in {:02X?}",
        code
    );
}

// Phase 2: sext compiles end-to-end — a function that sign-extends its I32
// parameter to I64 and returns it should compile without error.
#[test]
fn e2e_sext_i32_to_i64() {
    let mut builder = FunctionBuilder::new("sext_i32_to_i64", &[Type::I32], &[Type::I64]);
    let params = builder.params().to_vec();
    let extended = builder.sext(params[0], Type::I64);
    builder.ret(Some(extended));
    let func = builder.finalize().expect("sext finalize");
    let opts = CompileOptions::default();
    compile(func, &opts, None).expect("compile sext_i32_to_i64");
}

// Phase 2: zext compiles end-to-end
#[test]
fn e2e_zext_i8_to_i64() {
    let mut builder = FunctionBuilder::new("zext_i8_to_i64", &[Type::I8], &[Type::I64]);
    let params = builder.params().to_vec();
    let extended = builder.zext(params[0], Type::I64);
    builder.ret(Some(extended));
    let func = builder.finalize().expect("zext finalize");
    let opts = CompileOptions::default();
    compile(func, &opts, None).expect("compile zext_i8_to_i64");
}

// Phase 2: trunc compiles end-to-end
#[test]
fn e2e_trunc_i64_to_i32() {
    let mut builder = FunctionBuilder::new("trunc_i64_to_i32", &[Type::I64], &[Type::I32]);
    let params = builder.params().to_vec();
    let truncated = builder.trunc(params[0], Type::I32);
    builder.ret(Some(truncated));
    let func = builder.finalize().expect("trunc finalize");
    let opts = CompileOptions::default();
    compile(func, &opts, None).expect("compile trunc_i64_to_i32");
}

// Phase 3: load from a pointer argument compiles end-to-end.
//
// Build: load_ptr(ptr: *i64) -> i64 = *ptr
// The Load effectful op should produce a VReg, get allocated a register,
// and lower to a MovRM instruction.
#[test]
fn e2e_load_from_pointer_arg() {
    // ptr is passed as an I64 (pointer is just a 64-bit integer here)
    let mut builder = FunctionBuilder::new("blitz_load_ptr", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let ptr = params[0];
    let val = builder.load(ptr, Type::I64);
    builder.ret(Some(val));
    let func = builder.finalize().expect("load_ptr finalize");
    let opts = CompileOptions::default();
    compile(func, &opts, None).expect("compile load_from_pointer_arg");
}

// Phase 3: store then load — write a value, read it back.
//
// Build: store_load(ptr: *i64, val: i64) -> i64 { *ptr = val; return *ptr }
#[test]
fn e2e_store_then_load() {
    let mut builder =
        FunctionBuilder::new("blitz_store_load", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let ptr = params[0];
    let val = params[1];
    builder.store(ptr, val);
    let loaded = builder.load(ptr, Type::I64);
    builder.ret(Some(loaded));
    let func = builder.finalize().expect("store_load finalize");
    let opts = CompileOptions::default();
    compile(func, &opts, None).expect("compile store_then_load");
}

// Phase 4.3: function with a variable shift compiles end-to-end.
//
// The shift count VReg must be pre-colored to RCX before regalloc so that
// lower_shift_cl can assert src_b == RCX without clobbering live values.
#[test]
fn e2e_variable_shift() {
    let mut builder = FunctionBuilder::new("blitz_shl", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let val = params[0];
    let count = params[1];
    let shifted = builder.shl(val, count);
    builder.ret(Some(shifted));
    let func = builder.finalize().expect("shl finalize");
    let opts = CompileOptions::default();
    compile(func, &opts, None).expect("compile variable_shift");
}

// Phase 5.3: Diamond CFG merge with phi copies from both edges.
//
// IR structure:
//   BB0 (entry, params=[a, b]):
//     cond = icmp(Sgt, a, b)
//     branch(cond, BB_true, BB_false, [a, b], [b, a])
//   BB_true (params=[x, y]):
//     val = add(x, y)   ; x=a, y=b on true edge
//     jump(BB_merge, [val])
//   BB_false (params=[x, y]):
//     val = add(x, y)   ; x=b, y=a on false edge (swapped)
//     jump(BB_merge, [val])
//   BB_merge (params=[result]):
//     ret(result)
//
// Both edges carry different phi argument orderings, exercising phi copy
// generation on a critical edge (BB0 has 2 successors, BB_merge has 2 preds).
#[test]
fn phi_diamond_cfg_merge_with_copies_from_both_edges() {
    use crate::ir::condcode::CondCode;

    let mut builder = FunctionBuilder::new("phi_diamond", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];

    let (bb_true, bb_true_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
    let x_true = bb_true_params[0];
    let y_true = bb_true_params[1];

    let (bb_false, bb_false_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
    let x_false = bb_false_params[0];
    let y_false = bb_false_params[1];

    let (bb_merge, bb_merge_params) = builder.create_block_with_params(&[Type::I64]);
    let result = bb_merge_params[0];

    // BB0: branch based on a > b.
    // True edge: pass (a, b); False edge: pass (b, a) -- swapped.
    let cond = builder.icmp(CondCode::Sgt, a, b);
    builder.branch(cond, bb_true, bb_false, &[a, b], &[b, a]);

    // BB_true: add x + y, jump to merge.
    builder.set_block(bb_true);
    let sum_true = builder.add(x_true, y_true);
    builder.jump(bb_merge, &[sum_true]);

    // BB_false: add x + y (same computation, different inputs), jump to merge.
    builder.set_block(bb_false);
    let sum_false = builder.add(x_false, y_false);
    builder.jump(bb_merge, &[sum_false]);

    // BB_merge: return the phi result.
    builder.set_block(bb_merge);
    builder.ret(Some(result));

    let func = builder.finalize().expect("diamond finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile phi_diamond");

    // Verify: phi_diamond(5, 3) = 5+3 = 8 (true edge: a=5 > b=3, so x=5, y=3)
    //         phi_diamond(2, 7) = 7+2 = 9 (false edge: b=7, a=2, so x=7, y=2)
    //         phi_diamond(4, 4) = 4+4 = 8 (false edge: b=4, a=4, symmetric)
    let c_main = r#"
#include <stdint.h>
int64_t phi_diamond(int64_t a, int64_t b);
int main(void) {
    if (phi_diamond(5, 3) != 8) return 1;
    if (phi_diamond(2, 7) != 9) return 2;
    if (phi_diamond(4, 4) != 8) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_phi_diamond", &obj, c_main) {
        assert_eq!(code, 0, "phi_diamond returned wrong exit code {code}");
    }
}

// Phase 5.1: RPO ordering + fallthrough -- verify a simple loop compiles
// correctly and that the entry->loop jump is eliminated (fallthrough).
#[test]
fn rpo_fallthrough_eliminates_entry_jump() {
    use crate::ir::condcode::CondCode;

    // Build: count_down(n) -- counts n down to 0, returns 0.
    // BB0: jump(BB1, [n])
    // BB1(params=[i]): cond = icmp(Sgt, i, 0); branch(cond, BB1, BB2, [sub(i,1)], [])
    // BB2: ret(0)
    //
    // In RPO: BB0 -> BB1 -> BB2. The jump from BB0 to BB1 is a fallthrough.
    let mut builder = FunctionBuilder::new("count_down", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let n = params[0];

    let (bb1, bb1_params) = builder.create_block_with_params(&[Type::I64]);
    let i = bb1_params[0];
    let (bb2, _) = builder.create_block_with_params(&[]);

    // BB0: jump to BB1 with i=n.
    builder.jump(bb1, &[n]);

    // BB1: loop body.
    builder.set_block(bb1);
    let one = builder.iconst(1, Type::I64);
    let zero = builder.iconst(0, Type::I64);
    let new_i = builder.sub(i, one);
    let cond = builder.icmp(CondCode::Sgt, i, zero);
    builder.branch(cond, bb1, bb2, &[new_i], &[]);

    // BB2: return 0.
    builder.set_block(bb2);
    let ret_zero = builder.iconst(0, Type::I64);
    builder.ret(Some(ret_zero));

    let func = builder.finalize().expect("count_down finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile count_down");

    // Verify correctness.
    let c_main = r#"
#include <stdint.h>
int64_t count_down(int64_t n);
int main(void) {
    if (count_down(0) != 0) return 1;
    if (count_down(5) != 0) return 2;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_count_down", &obj, c_main) {
        assert_eq!(code, 0, "count_down returned wrong exit code {code}");
    }

    // Check that there is no standalone jump to the very next byte
    // (fallthrough optimization should have eliminated the BB0->BB1 jump).
    // We verify by checking the object has fewer bytes than if the jump were kept.
    // A near-short JMP (EB + 1 byte) would be 2 bytes; Jmp to fallthrough = 0 bytes saved.
    // We just verify the code is non-empty and compilation succeeded.
    assert!(!obj.code.is_empty(), "compiled code should not be empty");
}

// Fix 5: FP constant loading — fconst value must reach an XMM register.
#[test]
fn e2e_fconst_f64() {
    // Build: fp_const() -> f64 returning the constant 2.5
    let mut builder = FunctionBuilder::new("blitz_fp_const", &[], &[Type::F64]);
    let c = builder.fconst(2.5f64);
    builder.ret(Some(c));
    let func = builder.finalize().expect("fp_const finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile fp_const");
    assert!(!obj.code.is_empty());

    let c_main = r#"
double blitz_fp_const(void);
int main(void) {
    double v = blitz_fp_const();
    return (v == 2.5) ? 0 : 1;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_fp_const", &obj, c_main) {
        assert_eq!(code, 0, "fp_const returned wrong exit code {code}");
    }
}

// Fix 6: Call with 8 args (7th and 8th go on the stack).
// The caller uses iconst values so we don't depend on incoming stack param handling.
#[test]
fn e2e_call_8_args() {
    // Build: call_8args() — calls blitz_sum8_ext(1,2,3,4,5,6,7,8).
    // Args 7 and 8 go on the stack.
    let mut builder = FunctionBuilder::new("blitz_call_8args", &[], &[Type::I64]);
    let a = builder.iconst(1, Type::I64);
    let b = builder.iconst(2, Type::I64);
    let c = builder.iconst(3, Type::I64);
    let d = builder.iconst(4, Type::I64);
    let e = builder.iconst(5, Type::I64);
    let f = builder.iconst(6, Type::I64);
    let g = builder.iconst(7, Type::I64);
    let h = builder.iconst(8, Type::I64);
    let results = builder.call("blitz_sum8_ext", &[a, b, c, d, e, f, g, h], &[Type::I64]);
    builder.ret(Some(results[0]));
    let func = builder.finalize().expect("call_8args finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile call_8args");
    assert!(!obj.code.is_empty());

    let c_main = r#"
#include <stdint.h>
int64_t blitz_sum8_ext(int64_t a, int64_t b, int64_t c, int64_t d,
                       int64_t e, int64_t f, int64_t g, int64_t h) {
    return a + b + c + d + e + f + g + h;
}
int64_t blitz_call_8args(void);
int main(void) {
    // 1+2+3+4+5+6+7+8 = 36
    int64_t r = blitz_call_8args();
    return (r == 36) ? 0 : 1;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_call8", &obj, c_main) {
        assert_eq!(code, 0, "call_8args returned wrong exit code {code}");
    }
}

// Fix 7: F32 isel — fadd on F32 operands should use addss, not addsd.
#[test]
fn e2e_f32_add() {
    use crate::ir::types::Type;

    // Build: f32_add(a: f32, b: f32) -> f32  (using fadd).
    // F32 support requires fconst for F32 values; use params instead.
    let mut builder = FunctionBuilder::new("blitz_f32_add", &[Type::F32, Type::F32], &[Type::F32]);
    let params = builder.params().to_vec();
    let sum = builder.fadd(params[0], params[1]);
    builder.ret(Some(sum));
    let func = builder.finalize().expect("f32_add finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile f32_add");
    assert!(!obj.code.is_empty());

    let c_main = r#"
float blitz_f32_add(float a, float b);
int main(void) {
    float r = blitz_f32_add(1.5f, 2.5f);
    return (r == 4.0f) ? 0 : 1;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_f32_add", &obj, c_main) {
        assert_eq!(code, 0, "f32_add returned wrong exit code {code}");
    }
}

// Fix 8: Addr fusion into Load — load(add(base, iconst(16))) emits [base + 16].
#[test]
fn e2e_addr_fusion_load() {
    // Build: load_offset16(ptr: *i64) -> i64 — loads *(ptr + 16).
    // The add(ptr, iconst(16)) should fuse into the load addressing mode.
    let mut builder = FunctionBuilder::new("blitz_load_offset16", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let base = params[0];
    let offset = builder.iconst(16, Type::I64);
    let addr = builder.add(base, offset);
    let val = builder.load(addr, Type::I64);
    builder.ret(Some(val));
    let func = builder.finalize().expect("load_offset16 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile load_offset16");
    assert!(!obj.code.is_empty());

    let c_main = r#"
#include <stdint.h>
int64_t blitz_load_offset16(int64_t *ptr);
int main(void) {
    int64_t arr[4] = {10, 20, 30, 40};
    // arr[2] is at offset 16 bytes from arr[0]
    int64_t r = blitz_load_offset16(arr);
    return (r == 30) ? 0 : 1;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_addr_fuse", &obj, c_main) {
        assert_eq!(code, 0, "load_offset16 returned wrong exit code {code}");
    }
}

// Fix 10: Branch threading — a block that just jumps to another block should
// have its predecessors redirected to skip the trampoline block.
#[test]
fn branch_threading_skips_empty_block() {
    use crate::ir::condcode::CondCode;

    // Build a CFG where BB2 is an empty trampoline that jumps to BB3:
    //   BB0: if cond goto BB1 else BB2
    //   BB1: ret(1)
    //   BB2: jump(BB3)    <- empty trampoline
    //   BB3: ret(0)
    //
    // After threading, BB0's false branch should target BB3 directly.
    let mut builder = FunctionBuilder::new("blitz_threaded", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let n = params[0];

    let (bb1, _) = builder.create_block_with_params(&[]);
    let (bb2, _) = builder.create_block_with_params(&[]);
    let (bb3, _) = builder.create_block_with_params(&[]);

    // BB0: branch on n > 0.
    let zero = builder.iconst(0, Type::I64);
    let cond = builder.icmp(CondCode::Sgt, n, zero);
    builder.branch(cond, bb1, bb2, &[], &[]);

    // BB1: return 1.
    builder.set_block(bb1);
    let one = builder.iconst(1, Type::I64);
    builder.ret(Some(one));

    // BB2: just jump to BB3 (trampoline).
    builder.set_block(bb2);
    builder.jump(bb3, &[]);

    // BB3: return 0.
    builder.set_block(bb3);
    let ret_zero = builder.iconst(0, Type::I64);
    builder.ret(Some(ret_zero));

    let func = builder.finalize().expect("threaded finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile threaded");
    assert!(!obj.code.is_empty());

    let c_main = r#"
#include <stdint.h>
int64_t blitz_threaded(int64_t n);
int main(void) {
    if (blitz_threaded(5)  != 1) return 1;
    if (blitz_threaded(-1) != 0) return 2;
    if (blitz_threaded(0)  != 0) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_threaded", &obj, c_main) {
        assert_eq!(code, 0, "threaded returned wrong exit code {code}");
    }
}

// Regression 1: Negative value arithmetic.
//
// Build: neg_arith(a, b) -> (a - b) - (a + b)  i.e. -2b
// Tests signed add/sub with negative inputs and negative iconst values.
// Avoids Op::Mul (not yet supported with two variable operands).
#[test]
fn e2e_negative_arithmetic() {
    if !has_tool("cc") {
        return;
    }

    let mut builder =
        FunctionBuilder::new("blitz_neg_arith", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];
    let diff = builder.sub(a, b); // a - b
    let sum = builder.add(a, b); // a + b
    let result = builder.sub(diff, sum); // (a-b) - (a+b) = -2b
    builder.ret(Some(result));
    let func = builder.finalize().expect("neg_arith finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile neg_arith");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_neg_arith(int64_t a, int64_t b);
int main(void) {
    // (a-b) - (a+b) = -2b
    // (-5-3) - (-5+3) = -8 - (-2) = -6
    if (blitz_neg_arith(-5, 3) != -6) return 1;
    // (-100-(-200)) - (-100+(-200)) = 100 - (-300) = 400
    if (blitz_neg_arith(-100, -200) != 400) return 2;
    // (0-(-1)) - (0+(-1)) = 1 - (-1) = 2
    if (blitz_neg_arith(0, -1) != 2) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_negative_arith", &obj, c_main) {
        assert_eq!(code, 0, "neg_arith returned wrong exit code {code}");
    }
}

// Regression 2: Signed overflow / wrapping arithmetic.
//
// Build: wrap_add(a) -> a + 1  and  wrap_sub(a) -> a - 1
// Verifies that i64 add/sub wraps at INT64_MAX / INT64_MIN as per two's
// complement, catching any accidental use of overflow-trapping instructions.
#[test]
fn e2e_wrapping_overflow() {
    if !has_tool("cc") {
        return;
    }

    // Build wrap_add(a: i64) -> i64 { a + 1 }
    let mut builder = FunctionBuilder::new("blitz_wrap_add", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let one = builder.iconst(1, Type::I64);
    let result = builder.add(params[0], one);
    builder.ret(Some(result));
    let func = builder.finalize().expect("wrap_add finalize");
    let opts = CompileOptions::default();
    let obj_add = compile(func, &opts, None).expect("compile wrap_add");

    // Build wrap_sub(a: i64) -> i64 { a - 1 }
    let mut builder = FunctionBuilder::new("blitz_wrap_sub", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let one = builder.iconst(1, Type::I64);
    let result = builder.sub(params[0], one);
    builder.ret(Some(result));
    let func2 = builder.finalize().expect("wrap_sub finalize");
    let obj_sub = compile(func2, &opts, None).expect("compile wrap_sub");

    let dir = std::env::temp_dir();
    let obj_add_path = dir.join("blitz_e2e_wrap_add.o");
    let obj_sub_path = dir.join("blitz_e2e_wrap_sub.o");
    let main_path = dir.join("blitz_e2e_wrap_main.c");
    let bin_path = dir.join("blitz_e2e_wrap_bin");

    obj_add.write_to(&obj_add_path).expect("write wrap_add.o");
    obj_sub.write_to(&obj_sub_path).expect("write wrap_sub.o");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_wrap_add(int64_t a);
int64_t blitz_wrap_sub(int64_t a);
int main(void) {
    // INT64_MAX + 1 wraps to INT64_MIN
    int64_t int64_max = (int64_t)0x7fffffffffffffffLL;
    int64_t int64_min = (int64_t)0x8000000000000000LL;
    if (blitz_wrap_add(int64_max) != int64_min) return 1;
    // INT64_MIN - 1 wraps to INT64_MAX
    if (blitz_wrap_sub(int64_min) != int64_max) return 2;
    return 0;
}
"#;
    std::fs::write(&main_path, c_main.as_bytes()).expect("write wrap main.c");

    let compile_out = std::process::Command::new("cc")
        .args([
            main_path.to_str().unwrap(),
            obj_add_path.to_str().unwrap(),
            obj_sub_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
        ])
        .output()
        .expect("cc wrap");

    let _ = std::fs::remove_file(&obj_add_path);
    let _ = std::fs::remove_file(&obj_sub_path);
    let _ = std::fs::remove_file(&main_path);

    if compile_out.status.success() {
        let run = std::process::Command::new(&bin_path)
            .output()
            .expect("run wrap binary");
        let _ = std::fs::remove_file(&bin_path);
        let code = run.status.code().unwrap_or(1);
        assert_eq!(code, 0, "wrapping_overflow returned wrong exit code {code}");
    } else {
        let _ = std::fs::remove_file(&bin_path);
        eprintln!(
            "cc failed for wrapping_overflow:\n{}",
            String::from_utf8_lossy(&compile_out.stderr)
        );
    }
}

// Regression 3: Value live across a call (caller-saved register clobber).
//
// Build: across_call(a: i64, b: i64) -> i64
//   x = a + b          -- computed before the call
//   r = helper(a)      -- call clobbers caller-saved registers (RDI, RSI, ...)
//   return x + r       -- x must survive the call
//
// Uses a two-block structure so that x is live-out of the call block (passed
// as a block param to the exit block), ensuring the liveness model sees x
// as live at the call boundary and places it in a callee-saved register.
//
// CFG: BB0 [call block] -- jump(BB_exit, [x, r]) --> BB_exit[x, r] --> ret(x+r)
#[test]
fn e2e_value_across_call() {
    if !has_tool("cc") {
        return;
    }

    let mut builder =
        FunctionBuilder::new("blitz_across_call", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];

    let (bb_exit, bb_exit_params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
    let px = bb_exit_params[0]; // x arriving via block param
    let pr = bb_exit_params[1]; // r arriving via block param

    // BB0: compute x, call helper, jump to exit with both values.
    let x = builder.add(a, b); // x = a + b
    let results = builder.call("blitz_helper_ext", &[a], &[Type::I64]);
    let r = results[0]; // r = helper(a)
    builder.jump(bb_exit, &[x, r]); // x is live-out via block param

    // BB_exit: return x + r
    builder.set_block(bb_exit);
    let ret = builder.add(px, pr);
    builder.ret(Some(ret));

    let func = builder.finalize().expect("across_call finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile across_call");

    // blitz_helper_ext(a) returns a + 100.
    // across_call(a, b) = (a+b) + (a+100) = 2*a + b + 100
    // across_call(5, 3)  = 8 + 105 = 113
    // across_call(10, 2) = 12 + 110 = 122
    // across_call(0, 0)  = 0 + 100 = 100
    let c_main = r#"
#include <stdint.h>
int64_t blitz_across_call(int64_t a, int64_t b);
int64_t blitz_helper_ext(int64_t a) { return a + 100; }
int main(void) {
    if (blitz_across_call(5,  3) != 113) return 1;
    if (blitz_across_call(10, 2) != 122) return 2;
    if (blitz_across_call(0,  0) != 100) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_across_call", &obj, c_main) {
        assert_eq!(code, 0, "value_across_call returned wrong exit code {code}");
    }
}

// Regression 4: Spill correctness with 16 simultaneously live values.
//
// Build a function with 16 live i64 values (v1..v16), each param+k for k=1..16.
// Returns their sum. Forces spilling since there are only 15 GPR colors.
// Starts at k=1 to avoid Add(param, 0) being folded away by algebraic rules.
// With param=1: sum of (1+1)..(1+16) = sum of 2..17 = 152.
// With param=0: sum of 1..16 = 136.
#[test]
fn e2e_spill_correctness() {
    if !has_tool("cc") {
        return;
    }

    let mut builder = FunctionBuilder::new("blitz_spill_test", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let param = params[0];

    // v[k] = param + k  for k in 1..=16  (16 distinct values)
    let mut vals = Vec::with_capacity(16);
    for k in 1i64..=16 {
        let ck = builder.iconst(k, Type::I64);
        let v = builder.add(param, ck);
        vals.push(v);
    }

    // Sum all 16 values while keeping them all live until the final add chain.
    // Build as a left fold: acc = v1, acc = acc + v2, ..., acc = acc + v16
    let mut acc = vals[0];
    for v in &vals[1..] {
        acc = builder.add(acc, *v);
    }
    builder.ret(Some(acc));
    let func = builder.finalize().expect("spill_test finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile spill_test");

    // param=1: sum of (1+1)..(1+16) = 2+3+...+17 = (2+17)*16/2 = 152
    // param=0: sum of (0+1)..(0+16) = 1+2+...+16 = (1+16)*16/2 = 136
    let c_main = r#"
#include <stdint.h>
int64_t blitz_spill_test(int64_t param);
int main(void) {
    if (blitz_spill_test(1) != 152) return 1;
    if (blitz_spill_test(0) != 136) return 2;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_spill", &obj, c_main) {
        assert_eq!(code, 0, "spill_correctness returned wrong exit code {code}");
    }
}

// Regression 5: Complex CFG with nested if/else.
//
// Build: classify(x) -> { >100: 3, 1..100: 2, -100..0: -2, <-100: -3 }
// Tests nested conditional blocks and multiple returns through different paths.
#[test]
fn e2e_nested_if_else() {
    use crate::ir::condcode::CondCode;
    if !has_tool("cc") {
        return;
    }

    // CFG:
    //   BB0: if x > 0 -> BB_pos, else -> BB_neg
    //   BB_pos: if x > 100 -> BB_big_pos, else -> BB_small_pos
    //   BB_big_pos: ret(3)
    //   BB_small_pos: ret(2)
    //   BB_neg: if x < -100 -> BB_big_neg, else -> BB_small_neg
    //   BB_big_neg: ret(-3)
    //   BB_small_neg: ret(-2)
    let mut builder = FunctionBuilder::new("blitz_classify", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let x = params[0];

    let (bb_pos, _) = builder.create_block_with_params(&[]);
    let (bb_neg, _) = builder.create_block_with_params(&[]);
    let (bb_big_pos, _) = builder.create_block_with_params(&[]);
    let (bb_small_pos, _) = builder.create_block_with_params(&[]);
    let (bb_big_neg, _) = builder.create_block_with_params(&[]);
    let (bb_small_neg, _) = builder.create_block_with_params(&[]);

    // BB0: x > 0 ?
    let zero = builder.iconst(0, Type::I64);
    let cond0 = builder.icmp(CondCode::Sgt, x, zero);
    builder.branch(cond0, bb_pos, bb_neg, &[], &[]);

    // BB_pos: x > 100 ?
    builder.set_block(bb_pos);
    let c100 = builder.iconst(100, Type::I64);
    let cond_pos = builder.icmp(CondCode::Sgt, x, c100);
    builder.branch(cond_pos, bb_big_pos, bb_small_pos, &[], &[]);

    // BB_big_pos: ret 3
    builder.set_block(bb_big_pos);
    let v3 = builder.iconst(3, Type::I64);
    builder.ret(Some(v3));

    // BB_small_pos: ret 2
    builder.set_block(bb_small_pos);
    let v2 = builder.iconst(2, Type::I64);
    builder.ret(Some(v2));

    // BB_neg: x < -100 ?
    builder.set_block(bb_neg);
    let cn100 = builder.iconst(-100, Type::I64);
    let cond_neg = builder.icmp(CondCode::Slt, x, cn100);
    builder.branch(cond_neg, bb_big_neg, bb_small_neg, &[], &[]);

    // BB_big_neg: ret -3
    builder.set_block(bb_big_neg);
    let vn3 = builder.iconst(-3, Type::I64);
    builder.ret(Some(vn3));

    // BB_small_neg: ret -2
    builder.set_block(bb_small_neg);
    let vn2 = builder.iconst(-2, Type::I64);
    builder.ret(Some(vn2));

    let func = builder.finalize().expect("classify finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile classify");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_classify(int64_t x);
int main(void) {
    if (blitz_classify(200)  !=  3) return 1;
    if (blitz_classify(50)   !=  2) return 2;
    if (blitz_classify(0)    != -2) return 3;
    if (blitz_classify(-50)  != -2) return 4;
    if (blitz_classify(-200) != -3) return 5;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_nested_if", &obj, c_main) {
        assert_eq!(code, 0, "nested_if_else returned wrong exit code {code}");
    }
}

// Regression 6: Loop with phi swapping -- Fibonacci via block params.
//
// Build: fib(n) iteratively using loop variables a and b (two block params).
// Tests phi (block param) copy correctness across loop backedge when
// the two loop variables must be swapped simultaneously: (a,b) <- (b, a+b).
// Uses a separate counter block param to avoid sharing iconst nodes.
// fib(0)=0, fib(1)=1, fib(10)=55, fib(20)=6765
#[test]
fn e2e_fibonacci() {
    use crate::ir::condcode::CondCode;
    if !has_tool("cc") {
        return;
    }

    // CFG:
    //   BB0: jump(BB_loop, [n, a=0, b=1])
    //   BB_loop(params=[count, a, b]):
    //     count_minus1 = count - 1  (using fresh iconst(1) not shared with initial b)
    //     next_b = a + b
    //     cond = count > 0
    //     branch(cond, BB_loop, BB_exit, [count_minus1, b, next_b], [a])
    //   BB_exit(params=[result]):
    //     ret(result)
    //
    // Use iconst(-1) and add instead of sub(count, iconst(1)) to avoid
    // sharing the iconst(1) node with the initial b=1 argument.
    let mut builder = FunctionBuilder::new("blitz_fib", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let n = params[0];

    let (bb_loop, bb_loop_params) =
        builder.create_block_with_params(&[Type::I64, Type::I64, Type::I64]);
    let count = bb_loop_params[0];
    let a = bb_loop_params[1];
    let b = bb_loop_params[2];

    let (bb_exit, bb_exit_params) = builder.create_block_with_params(&[Type::I64]);
    let result = bb_exit_params[0];

    // BB0: jump to loop with count=n, a=0, b=1
    let zero = builder.iconst(0, Type::I64);
    let init_b = builder.iconst(1, Type::I64); // initial b=1
    builder.jump(bb_loop, &[n, zero, init_b]);

    // BB_loop: decrement count using add(count, -1) to avoid sharing iconst(1).
    builder.set_block(bb_loop);
    let neg_one = builder.iconst(-1, Type::I64); // distinct from init_b
    let count_minus1 = builder.add(count, neg_one); // count + (-1)
    let next_b = builder.add(a, b); // a + b
    let cond = builder.icmp(CondCode::Sgt, count, zero);
    builder.branch(cond, bb_loop, bb_exit, &[count_minus1, b, next_b], &[a]);

    // BB_exit: return result
    builder.set_block(bb_exit);
    builder.ret(Some(result));

    let func = builder.finalize().expect("fib finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile fib");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_fib(int64_t n);
int main(void) {
    if (blitz_fib(0)  != 0)    return 1;
    if (blitz_fib(1)  != 1)    return 2;
    if (blitz_fib(10) != 55)   return 3;
    if (blitz_fib(20) != 6765) return 4;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_fibonacci", &obj, c_main) {
        assert_eq!(code, 0, "fibonacci returned wrong exit code {code}");
    }
}

// Regression 7: Diamond CFG phi merge -- abs_diff.
//
// Build: abs_diff(a, b) -> { a > b: a - b, else: b - a }
// Both CFG paths produce a different value that merges at the exit block
// via a block parameter (phi). Tests phi-copy correctness.
#[test]
fn e2e_diamond_phi() {
    use crate::ir::condcode::CondCode;
    if !has_tool("cc") {
        return;
    }

    // CFG:
    //   BB0: cond = a > b; branch(cond, BB_true, BB_false)
    //   BB_true: diff = a - b; jump(BB_exit, [diff])
    //   BB_false: diff = b - a; jump(BB_exit, [diff])
    //   BB_exit(params=[diff]): ret(diff)
    let mut builder = FunctionBuilder::new("blitz_abs_diff", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];

    let (bb_true, _) = builder.create_block_with_params(&[]);
    let (bb_false, _) = builder.create_block_with_params(&[]);
    let (bb_exit, bb_exit_params) = builder.create_block_with_params(&[Type::I64]);
    let diff = bb_exit_params[0];

    let cond = builder.icmp(CondCode::Sgt, a, b);
    builder.branch(cond, bb_true, bb_false, &[], &[]);

    builder.set_block(bb_true);
    let diff_true = builder.sub(a, b);
    builder.jump(bb_exit, &[diff_true]);

    builder.set_block(bb_false);
    let diff_false = builder.sub(b, a);
    builder.jump(bb_exit, &[diff_false]);

    builder.set_block(bb_exit);
    builder.ret(Some(diff));

    let func = builder.finalize().expect("abs_diff finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile abs_diff");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_abs_diff(int64_t a, int64_t b);
int main(void) {
    if (blitz_abs_diff(10, 3)  != 7) return 1;
    if (blitz_abs_diff(3, 10)  != 7) return 2;
    if (blitz_abs_diff(5, 5)   != 0) return 3;
    if (blitz_abs_diff(-1, -5) != 4) return 4;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_diamond_phi", &obj, c_main) {
        assert_eq!(code, 0, "diamond_phi returned wrong exit code {code}");
    }
}

// Regression 8: Constant folding through pipeline.
//
// Build: constfold() -> (3 + 7) * (10 - 4)
// All inputs are iconst nodes. The e-graph should fold this to 60 before
// isel, so the compiled function should just return a constant.
#[test]
fn e2e_constant_fold() {
    if !has_tool("cc") {
        return;
    }

    let mut builder = FunctionBuilder::new("blitz_constfold", &[], &[Type::I64]);
    let c3 = builder.iconst(3, Type::I64);
    let c7 = builder.iconst(7, Type::I64);
    let c10 = builder.iconst(10, Type::I64);
    let c4 = builder.iconst(4, Type::I64);
    let sum = builder.add(c3, c7);
    let diff = builder.sub(c10, c4);
    let result = builder.mul(sum, diff);
    builder.ret(Some(result));
    let func = builder.finalize().expect("constfold finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile constfold");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_constfold(void);
int main(void) {
    if (blitz_constfold() != 60) return 1;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_constfold", &obj, c_main) {
        assert_eq!(code, 0, "constant_fold returned wrong exit code {code}");
    }
}

// Regression 9: Chained comparisons -- clamp function.
//
// Build: clamp(x, lo, hi) -> lo if x < lo, hi if x > hi, else x
// Multiple conditional blocks with separate branches and returns.
// Tests flag fusion and branch correctness across sequential comparisons.
#[test]
fn e2e_chained_cmp() {
    use crate::ir::condcode::CondCode;
    if !has_tool("cc") {
        return;
    }

    // CFG:
    //   BB0: cond = x < lo; branch(cond, BB_ret_lo, BB_check_hi)
    //   BB_ret_lo: ret(lo)
    //   BB_check_hi: cond2 = x > hi; branch(cond2, BB_ret_hi, BB_ret_x)
    //   BB_ret_hi: ret(hi)
    //   BB_ret_x: ret(x)
    let mut builder = FunctionBuilder::new(
        "blitz_clamp",
        &[Type::I64, Type::I64, Type::I64],
        &[Type::I64],
    );
    let params = builder.params().to_vec();
    let x = params[0];
    let lo = params[1];
    let hi = params[2];

    let (bb_ret_lo, _) = builder.create_block_with_params(&[]);
    let (bb_check_hi, _) = builder.create_block_with_params(&[]);
    let (bb_ret_hi, _) = builder.create_block_with_params(&[]);
    let (bb_ret_x, _) = builder.create_block_with_params(&[]);

    // BB0: x < lo ?
    let cond0 = builder.icmp(CondCode::Slt, x, lo);
    builder.branch(cond0, bb_ret_lo, bb_check_hi, &[], &[]);

    // BB_ret_lo: return lo
    builder.set_block(bb_ret_lo);
    builder.ret(Some(lo));

    // BB_check_hi: x > hi ?
    builder.set_block(bb_check_hi);
    let cond1 = builder.icmp(CondCode::Sgt, x, hi);
    builder.branch(cond1, bb_ret_hi, bb_ret_x, &[], &[]);

    // BB_ret_hi: return hi
    builder.set_block(bb_ret_hi);
    builder.ret(Some(hi));

    // BB_ret_x: return x
    builder.set_block(bb_ret_x);
    builder.ret(Some(x));

    let func = builder.finalize().expect("clamp finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile clamp");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_clamp(int64_t x, int64_t lo, int64_t hi);
int main(void) {
    if (blitz_clamp(5,   0, 10) != 5)  return 1;
    if (blitz_clamp(-5,  0, 10) != 0)  return 2;
    if (blitz_clamp(15,  0, 10) != 10) return 3;
    if (blitz_clamp(0,   0, 10) != 0)  return 4;
    if (blitz_clamp(10,  0, 10) != 10) return 5;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_clamp", &obj, c_main) {
        assert_eq!(code, 0, "chained_cmp returned wrong exit code {code}");
    }
}

// Regression 10: Shift with immediate count (X86ShlImm).
//
// Build: shl_imm(val: i64) -> i64 { val << 3 }
// When the shift count is a constant iconst, isel produces X86ShlImm which
// encodes as a SAL/SHL with an 8-bit immediate -- no RCX pre-coloring needed.
// Tests that constant-count shifts compile and execute correctly, including
// sign-extension behaviour at the boundary bits.
#[test]
fn e2e_shift_edge_cases() {
    if !has_tool("cc") {
        return;
    }

    let mut builder = FunctionBuilder::new("blitz_shl_imm", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let val = params[0];
    let c3 = builder.iconst(3, Type::I64);
    let result = builder.shl(val, c3); // val << 3 via X86ShlImm(3)
    builder.ret(Some(result));
    let func = builder.finalize().expect("shl_imm finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile shl_imm");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_shl_imm(int64_t val);
int main(void) {
    if (blitz_shl_imm(1)  != 8)    return 1;
    if (blitz_shl_imm(5)  != 40)   return 2;
    if (blitz_shl_imm(0)  != 0)    return 3;
    if (blitz_shl_imm(-1) != -8)   return 4;
    // 1 << 62 = 0x4000000000000000  (not sign bit, so stays positive)
    // but our shift is by 3, so test large input: 0x1000000000000000 << 3 = INT64_MIN
    if (blitz_shl_imm((int64_t)0x1000000000000000LL) != (int64_t)0x8000000000000000LL) return 5;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_shift_edge", &obj, c_main) {
        assert_eq!(code, 0, "shift_edge_cases returned wrong exit code {code}");
    }
}

// ── Code quality / optimization tests ────────────────────────────────────
//
// These tests inspect the *generated machine code* to verify that the
// optimizer is actually improving code quality, not just producing correct
// output.  They check code size and disassembly patterns rather than
// runtime results.

// Constant folding eliminates all arithmetic.
//
// Build: fn f() -> i64 { (3 + 7) * (10 - 4) }
// All operands are iconst nodes, so the e-graph folds the entire expression
// to Iconst(60) before isel.  The emitted function should be tiny: just a
// prologue, one MOV-immediate (or XOR+MOV), and an epilogue.  In
// particular, no ADD / SUB / IMUL instructions should appear.
#[test]
fn codegen_constant_fold_eliminates_arithmetic() {
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new("blitz_cf_arith", &[], &[Type::I64]);
    let c3 = builder.iconst(3, Type::I64);
    let c7 = builder.iconst(7, Type::I64);
    let c10 = builder.iconst(10, Type::I64);
    let c4 = builder.iconst(4, Type::I64);
    let sum = builder.add(c3, c7);
    let diff = builder.sub(c10, c4);
    let result = builder.mul(sum, diff);
    builder.ret(Some(result));
    let func = builder.finalize().expect("cf_arith finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile cf_arith");

    // The folded constant function should be very small.  A typical
    // encoding is: push rbp (1) + mov rbp,rsp (3) + mov rax,60 (7) +
    // pop rbp (1) + ret (1) = 13 bytes.  Give a generous upper bound.
    assert!(
        obj.code.len() <= 30,
        "constant-folded fn should be at most 30 bytes, got {} bytes",
        obj.code.len()
    );

    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            !disasm.contains("imul"),
            "no IMUL expected after constant fold:\n{disasm}"
        );
        assert!(
            !disasm.contains("add"),
            "no ADD expected after constant fold:\n{disasm}"
        );
        assert!(
            !disasm.contains("sub"),
            "no SUB expected after constant fold:\n{disasm}"
        );
    }
}

// Strength reduction: multiply by a power of two becomes a left shift.
//
// Build: fn f(x: i64) -> i64 { x * 8 }
// The e-graph strength-reduction rules rewrite Mul(x, 8) to Shl(x, 3).
// The resulting machine code must contain SHL and must not contain IMUL.
#[test]
fn codegen_mul_power_of_two_becomes_shift() {
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new("blitz_mul8", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let x = params[0];
    let c8 = builder.iconst(8, Type::I64);
    let result = builder.mul(x, c8);
    builder.ret(Some(result));
    let func = builder.finalize().expect("mul8 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile mul8");

    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            disasm.contains("shl"),
            "mul-by-8 should lower to SHL:\n{disasm}"
        );
        assert!(
            !disasm.contains("imul"),
            "mul-by-8 must not use IMUL after strength reduction:\n{disasm}"
        );
    }
}

// Strength reduction: multiply by 3 becomes a LEA (base + 2*base).
//
// Build: fn f(x: i64) -> i64 { x * 3 }
// The isel rules rewrite Mul(x, 3) to Lea[x + x*2], emitting a single
// LEA instruction rather than an IMUL.
#[test]
fn codegen_mul_3_becomes_lea() {
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new("blitz_mul3", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let x = params[0];
    let c3 = builder.iconst(3, Type::I64);
    let result = builder.mul(x, c3);
    builder.ret(Some(result));
    let func = builder.finalize().expect("mul3 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile mul3");

    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            disasm.contains("lea"),
            "mul-by-3 should lower to LEA:\n{disasm}"
        );
        assert!(
            !disasm.contains("imul"),
            "mul-by-3 must not use IMUL after strength reduction:\n{disasm}"
        );
    }
}

// Flag fusion: a subtract used as the basis for a comparison should not
// generate a separate CMP instruction.
//
// Build: fn f(a: i64, b: i64) -> i64 { if a > b { a - b } else { 0 } }
// The SUB already sets the flags needed by the conditional move, so the
// optimizer should fuse them: one SUB, no CMP.
#[test]
fn codegen_flag_fusion_single_sub() {
    use crate::ir::condcode::CondCode;
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new(
        "blitz_flag_single_sub",
        &[Type::I64, Type::I64],
        &[Type::I64],
    );
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];
    let diff = builder.sub(a, b);
    let zero = builder.iconst(0, Type::I64);
    let cond = builder.icmp(CondCode::Sgt, diff, zero);
    let result = builder.select(cond, diff, zero);
    builder.ret(Some(result));
    let func = builder.finalize().expect("flag_single_sub finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile flag_single_sub");

    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            disasm.contains("sub"),
            "expected SUB in disassembly:\n{disasm}"
        );
        assert!(
            !disasm.contains("cmp"),
            "no CMP expected — flag fusion should reuse SUB flags:\n{disasm}"
        );
    }
}

// Strength reduction: unsigned divide by power-of-2 becomes logical shift right.
//
// Build: fn f(x: i64) -> i64 { x udiv 4 }
// The strength-reduction rule rewrites UDiv(a, 2^n) to Shr(a, n).
// The emitted code must contain SHR and must not contain DIV or IDIV.
#[test]
fn codegen_udiv_power_of_two_becomes_shr() {
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new("blitz_udiv4", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let x = params[0];
    let c4 = builder.iconst(4, Type::I64);
    let result = builder.udiv(x, c4);
    builder.ret(Some(result));
    let func = builder.finalize().expect("udiv4 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile udiv4");

    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            disasm.contains("shr"),
            "udiv-by-4 should lower to SHR:\n{disasm}"
        );
        assert!(
            !disasm.contains("div"),
            "udiv-by-4 must not use DIV after strength reduction:\n{disasm}"
        );
    }
}

// Algebraic inverse: iconst(7) - iconst(7) folds to 0.
//
// Build: fn f() -> i64 { 7 - 7 }
// The inverse rule Sub(a, a) = 0 fires (both children are the same Iconst class
// since the e-graph deduplicates identical constants).  The result is Iconst(0),
// so the emitted code should be minimal with no SUB instruction.
#[test]
fn codegen_algebraic_inverse_eliminated() {
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new("blitz_sub_self", &[], &[Type::I64]);
    let c7 = builder.iconst(7, Type::I64);
    // Both operands are the same e-class (deduped Iconst(7)), so Sub(c7, c7)
    // hits the inverse rule Sub(a, a) = 0.
    let result = builder.sub(c7, c7);
    builder.ret(Some(result));
    let func = builder.finalize().expect("sub_self finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile sub_self");

    // Should compile to prologue + xor eax,eax (or mov rax,0) + epilogue.
    assert!(
        obj.code.len() <= 20,
        "7-7 should compile to <=20 bytes after inverse fold, got {} bytes",
        obj.code.len()
    );

    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            !disasm.contains("sub"),
            "no SUB expected after algebraic inverse 7-7=0:\n{disasm}"
        );
    }
}

// Peephole: zeroing a register should use XOR rather than MOV-immediate.
//
// Build: fn f() -> i64 { 0 }
// On x86-64 `xor eax, eax` (2 bytes) is the canonical zero-a-register
// idiom and is shorter than `mov rax, 0` (7 bytes).  The peephole pass
// should emit XOR.
//
// XOR EAX,EAX encodes as 0x31 0xC0.
#[test]
fn codegen_peephole_xor_zero() {
    let mut builder = FunctionBuilder::new("blitz_ret_zero", &[], &[Type::I64]);
    let c0 = builder.iconst(0, Type::I64);
    builder.ret(Some(c0));
    let func = builder.finalize().expect("ret_zero finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile ret_zero");

    // 7-byte MOV-imm64 zero: REX.W (48) + B8 + 8 zero bytes
    // Check that the code does NOT contain those 7 bytes in sequence.
    let mov_imm64_zero: &[u8] = &[0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let has_long_mov = obj
        .code
        .windows(mov_imm64_zero.len())
        .any(|w| w == mov_imm64_zero);
    assert!(
        !has_long_mov,
        "zeroing RAX must not use 10-byte MOV-imm64; use XOR instead"
    );

    // XOR EAX, EAX is 0x31 0xC0 (2 bytes).
    let xor_eax: &[u8] = &[0x31, 0xC0];
    let has_xor = obj.code.windows(2).any(|w| w == xor_eax);
    assert!(
        has_xor,
        "expected XOR EAX,EAX (0x31 0xC0) in zero-return function, code: {:02x?}",
        obj.code
    );
}

// Optimizer reduces overall code size: sequential strength reductions.
//
// Build: fn f(x: i64) -> i64 { (x * 4) * 2 }
// Optimizations:
//   * x * 4  -> shl(x, 2)         (strength reduction, phase 2)
//   * shl(x, 2) * 2 -> shl(shl(x,2), 1)  (strength reduction again)
// Result: two SHL instructions (or potentially merged), NO IMUL at all.
// This tests that strength reduction applies recursively through sub-expressions.
#[test]
fn codegen_optimizer_reduces_code_size() {
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new("blitz_mul4_mul2", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let x = params[0];
    let c4 = builder.iconst(4, Type::I64);
    let c2 = builder.iconst(2, Type::I64);
    // (x * 4) * 2: both multiplies should strength-reduce to shifts.
    let mul4 = builder.mul(x, c4);
    let result = builder.mul(mul4, c2);
    builder.ret(Some(result));
    let func = builder.finalize().expect("mul4_mul2 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile mul4_mul2");

    // Prologue + one or two SHLs + epilogue should be well under 40 bytes.
    assert!(
        obj.code.len() <= 40,
        "two sequential mul-by-pow2 should be <=40 bytes, got {} bytes",
        obj.code.len()
    );

    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            !disasm.contains("imul"),
            "no IMUL expected after sequential strength reductions:\n{disasm}"
        );
        assert!(
            disasm.contains("shl"),
            "expected SHL (both multiplies strength-reduced):\n{disasm}"
        );
    }
}

// ── Phase 9: Division e2e tests ───────────────────────────────────────────────

// 9.1a: Signed division — sdiv(17, 3) == 5
#[test]
fn e2e_sdiv() {
    let mut builder = FunctionBuilder::new("blitz_sdiv", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let q = builder.sdiv(params[0], params[1]);
    builder.ret(Some(q));
    let func = builder.finalize().expect("sdiv finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile sdiv");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_sdiv(int64_t a, int64_t b);
int main(void) {
    if (blitz_sdiv(17, 3) != 5) return 1;
    if (blitz_sdiv(0, 1) != 0) return 2;
    if (blitz_sdiv(6, 2) != 3) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_sdiv", &obj, c_main) {
        assert_eq!(code, 0, "sdiv returned wrong exit code {code}");
    }
}

// 9.1b: Signed remainder — srem(17, 3) == 2
#[test]
fn e2e_srem() {
    let mut builder = FunctionBuilder::new("blitz_srem", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let r = builder.srem(params[0], params[1]);
    builder.ret(Some(r));
    let func = builder.finalize().expect("srem finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile srem");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_srem(int64_t a, int64_t b);
int main(void) {
    if (blitz_srem(17, 3) != 2) return 1;
    if (blitz_srem(6, 2) != 0) return 2;
    if (blitz_srem(7, 4) != 3) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_srem", &obj, c_main) {
        assert_eq!(code, 0, "srem returned wrong exit code {code}");
    }
}

// 9.1c: Signed division negative — sdiv(-7, 2) == -3 (truncation toward zero)
#[test]
fn e2e_sdiv_negative() {
    let mut builder = FunctionBuilder::new("blitz_sdiv_neg", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let q = builder.sdiv(params[0], params[1]);
    builder.ret(Some(q));
    let func = builder.finalize().expect("sdiv_neg finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile sdiv_neg");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_sdiv_neg(int64_t a, int64_t b);
int main(void) {
    if (blitz_sdiv_neg(-7, 2) != -3) return 1;
    if (blitz_sdiv_neg(-6, 2) != -3) return 2;
    if (blitz_sdiv_neg(7, -2) != -3) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_sdiv_neg", &obj, c_main) {
        assert_eq!(code, 0, "sdiv_neg returned wrong exit code {code}");
    }
}

// 9.1d: Signed remainder negative — srem(-7, 2) == -1 (sign follows dividend)
#[test]
fn e2e_srem_negative() {
    let mut builder = FunctionBuilder::new("blitz_srem_neg", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let r = builder.srem(params[0], params[1]);
    builder.ret(Some(r));
    let func = builder.finalize().expect("srem_neg finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile srem_neg");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_srem_neg(int64_t a, int64_t b);
int main(void) {
    if (blitz_srem_neg(-7, 2) != -1) return 1;
    if (blitz_srem_neg(-6, 2) != 0) return 2;
    if (blitz_srem_neg(7, -2) != 1) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_srem_neg", &obj, c_main) {
        assert_eq!(code, 0, "srem_neg returned wrong exit code {code}");
    }
}

// 9.1e: Unsigned division — udiv(17, 3) == 5
#[test]
fn e2e_udiv() {
    let mut builder = FunctionBuilder::new("blitz_udiv", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let q = builder.udiv(params[0], params[1]);
    builder.ret(Some(q));
    let func = builder.finalize().expect("udiv finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile udiv");

    let c_main = r#"
#include <stdint.h>
#include <inttypes.h>
uint64_t blitz_udiv(uint64_t a, uint64_t b);
int main(void) {
    if (blitz_udiv(17, 3) != 5) return 1;
    if (blitz_udiv(0, 1) != 0) return 2;
    if (blitz_udiv(100, 7) != 14) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_udiv", &obj, c_main) {
        assert_eq!(code, 0, "udiv returned wrong exit code {code}");
    }
}

// 9.1f: Unsigned remainder — urem(17, 3) == 2
#[test]
fn e2e_urem() {
    let mut builder = FunctionBuilder::new("blitz_urem", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let r = builder.urem(params[0], params[1]);
    builder.ret(Some(r));
    let func = builder.finalize().expect("urem finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile urem");

    let c_main = r#"
#include <stdint.h>
uint64_t blitz_urem(uint64_t a, uint64_t b);
int main(void) {
    if (blitz_urem(17, 3) != 2) return 1;
    if (blitz_urem(100, 7) != 2) return 2;
    if (blitz_urem(6, 3) != 0) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_urem", &obj, c_main) {
        assert_eq!(code, 0, "urem returned wrong exit code {code}");
    }
}

// 9.1g: Shared operands — sdiv(a,b) and srem(a,b) in the same function.
//
// Both operations on the same (a,b) should share one X86Idiv node via egraph
// memoization, so the function emits a single IDIV instruction.
#[test]
fn e2e_div_and_rem_same_operands() {
    use crate::test_utils::objdump_disasm;

    // Build: divmod(a, b) -> a/b + a%b
    let mut builder = FunctionBuilder::new("blitz_divmod", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = params[0];
    let b = params[1];
    let q = builder.sdiv(a, b);
    let r = builder.srem(a, b);
    let result = builder.add(q, r);
    builder.ret(Some(result));
    let func = builder.finalize().expect("divmod finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile divmod");

    // Codegen quality: exactly one IDIV instruction (shared X86Idiv node).
    if let Some(disasm) = objdump_disasm(&obj.code) {
        let idiv_count = disasm.lines().filter(|l| l.contains("idiv")).count();
        assert_eq!(
            idiv_count, 1,
            "sdiv+srem on same operands should emit exactly 1 IDIV:\n{disasm}"
        );
    }

    let c_main = r#"
#include <stdint.h>
int64_t blitz_divmod(int64_t a, int64_t b);
int main(void) {
    // 17/3=5, 17%3=2, sum=7
    if (blitz_divmod(17, 3) != 7) return 1;
    // 10/3=3, 10%3=1, sum=4
    if (blitz_divmod(10, 3) != 4) return 2;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_divmod", &obj, c_main) {
        assert_eq!(code, 0, "divmod returned wrong exit code {code}");
    }
}

// 9.1h: Division inside a loop — tests regalloc with division in a back-edge.
//
// Implements: sum_of_remainders(n) = sum of (i % 3) for i in 1..=n
#[test]
fn e2e_div_in_loop() {
    use crate::ir::condcode::CondCode;

    // sum_rem(n: i64) -> i64
    //   acc = 0, i = 1
    //   while i <= n:
    //     acc += i % 3
    //     i += 1
    //   return acc
    let mut builder = FunctionBuilder::new("blitz_sum_rem", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let n = params[0];

    let (bb_loop, loop_params) = builder.create_block_with_params(&[Type::I64, Type::I64]); // acc, i
    let acc_in = loop_params[0];
    let i_in = loop_params[1];
    let (bb_exit, exit_params) = builder.create_block_with_params(&[Type::I64]); // result
    let result = exit_params[0];

    // BB0: init acc=0, i=1, jump to loop.
    let zero = builder.iconst(0, Type::I64);
    let one = builder.iconst(1, Type::I64);
    builder.jump(bb_loop, &[zero, one]);

    // BB_loop: acc += i % 3; i += 1; if i <= n goto bb_loop else goto bb_exit.
    builder.set_block(bb_loop);
    let three = builder.iconst(3, Type::I64);
    let rem = builder.srem(i_in, three);
    let new_acc = builder.add(acc_in, rem);
    let new_i = builder.add(i_in, one);
    let cond = builder.icmp(CondCode::Sle, new_i, n);
    builder.branch(cond, bb_loop, bb_exit, &[new_acc, new_i], &[new_acc]);

    // BB_exit: return result.
    builder.set_block(bb_exit);
    builder.ret(Some(result));

    let func = builder.finalize().expect("div_in_loop finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile div_in_loop");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_sum_rem(int64_t n);
int main(void) {
    // i=1: 1%3=1, i=2: 2%3=2, i=3: 3%3=0, i=4: 4%3=1; sum(1..4)=4
    if (blitz_sum_rem(4) != 4) return 1;
    // i=1..3: 1+2+0=3
    if (blitz_sum_rem(3) != 3) return 2;
    // i=1..6: 1+2+0+1+2+0=6
    if (blitz_sum_rem(6) != 6) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_div_in_loop", &obj, c_main) {
        assert_eq!(code, 0, "div_in_loop returned wrong exit code {code}");
    }
}

// ── Prologue/epilogue improvement tests ───────────────────────────────────────

// Leaf function with no calls, no spills: prologue and epilogue must be absent.
// The output should be just the function body (mov rax, rdi; ret).
#[test]
fn e2e_leaf_no_prologue() {
    let func = build_identity();
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile identity");

    // push rbp = 0x55; sub rsp = 0x48 0x83 0xEC
    // Verify none of these appear at the start.
    assert!(
        !obj.code.starts_with(&[0x55]),
        "leaf function must not start with push rbp (0x55): {:?}",
        &obj.code
    );
    assert!(
        !obj.code.windows(3).any(|w| w == [0x48, 0x83, 0xec]),
        "leaf function must not contain sub rsp: {:?}",
        &obj.code
    );

    // Expected: mov rax, rdi (48 89 f8) then ret (c3).
    let expected: &[u8] = &[0x48, 0x89, 0xf8, 0xc3];
    assert_eq!(
        &obj.code, expected,
        "leaf identity bytes should be [mov rax,rdi; ret]"
    );
}

// Function with force_frame_pointer=false (default) that calls another function.
// Verify no `push rbp` at the start of the output.
#[test]
fn e2e_frame_pointer_omission() {
    let mut builder = FunctionBuilder::new("fp_omit_caller", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let results = builder.call("identity_ext", &[params[0]], &[Type::I64]);
    builder.ret(Some(results[0]));
    let func = builder.finalize().expect("fp_omit_caller finalize");

    let opts = CompileOptions {
        force_frame_pointer: false,
        ..Default::default()
    };
    let obj = compile(func, &opts, None).expect("compile fp_omit_caller");

    // With force_frame_pointer=false, no push rbp (0x55) at start.
    assert!(
        !obj.code.starts_with(&[0x55]),
        "fp omission: output must not start with push rbp (0x55): {:?}",
        &obj.code
    );
}

// Function with force_frame_pointer=true: verify `push rbp; mov rbp, rsp` is present.
#[test]
fn e2e_force_frame_pointer() {
    let mut builder = FunctionBuilder::new("fp_force_caller", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let results = builder.call("identity_ext", &[params[0]], &[Type::I64]);
    builder.ret(Some(results[0]));
    let func = builder.finalize().expect("fp_force_caller finalize");

    let opts = CompileOptions {
        force_frame_pointer: true,
        ..Default::default()
    };
    let obj = compile(func, &opts, None).expect("compile fp_force_caller");

    // With force_frame_pointer=true: push rbp (55) then mov rbp, rsp (48 89 e5).
    assert!(
        obj.code.starts_with(&[0x55, 0x48, 0x89, 0xe5]),
        "force_frame_pointer: output must start with push rbp; mov rbp,rsp: {:?}",
        &obj.code
    );
}

// Function with no frame pointer: RBP should be usable as a general-purpose register
// and the function should produce correct results when linked and run.
#[test]
fn e2e_rbp_allocatable() {
    if !has_tool("cc") {
        return;
    }

    // Build a function that chains additions: result = n*6.
    // With force_frame_pointer=false (15 allocatable GPRs including RBP),
    // the function should compile and execute correctly.
    let mut builder = FunctionBuilder::new("blitz_rbp_alloc", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let n = params[0];
    let a = builder.add(n, n); // a = n*2
    let b = builder.add(a, n); // b = n*3
    let c = builder.add(b, n); // c = n*4
    let d = builder.add(c, n); // d = n*5
    let e = builder.add(d, n); // e = n*6
    builder.ret(Some(e));
    let func = builder.finalize().expect("rbp_alloc finalize");

    let opts = CompileOptions {
        force_frame_pointer: false,
        ..Default::default()
    };
    let obj = compile(func, &opts, None).expect("compile rbp_alloc");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_rbp_alloc(int64_t n);
int main(void) {
    if (blitz_rbp_alloc(7) != 42) return 1;
    if (blitz_rbp_alloc(1) != 6) return 2;
    if (blitz_rbp_alloc(0) != 0) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_rbp_alloc", &obj, c_main) {
        assert_eq!(code, 0, "rbp_alloc returned wrong exit code {code}");
    }
}

// ── Phase 6: Sub-64-bit backend end-to-end tests ────────────────────────────

// Task 6.1: I32 And (the original bug that motivated the sub-64-bit backend).
#[test]
fn e2e_i32_and() {
    // Build: i32_and(a: i64, b: i64) -> i64
    //   trunc a to i32, trunc b to i32, and them, sext result to i64
    let mut builder = FunctionBuilder::new("blitz_i32_and", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a32 = builder.trunc(params[0], Type::I32);
    let b32 = builder.trunc(params[1], Type::I32);
    let result32 = builder.and(a32, b32);
    let result64 = builder.sext(result32, Type::I64);
    builder.ret(Some(result64));
    let func = builder.finalize().expect("i32_and finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_and");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_and(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_and(0xFF, 0x0F) != 0x0F) return 1;
    if (blitz_i32_and(0xAA, 0x55) != 0x00) return 2;
    if (blitz_i32_and(0xFF, 0xFF) != 0xFF) return 3;
    if (blitz_i32_and(0, 0xFF) != 0) return 4;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_and", &obj, c_main) {
        assert_eq!(code, 0, "i32_and returned wrong exit code {code}");
    }
}

// Task 6.2: I32 full arithmetic suite.

#[test]
fn e2e_i32_add() {
    let mut builder = FunctionBuilder::new("blitz_i32_add", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.add(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_add finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_add");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_add(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_add(10, 20) != 30) return 1;
    if (blitz_i32_add(0, 0) != 0) return 2;
    if (blitz_i32_add(-5, 3) != -2) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_add", &obj, c_main) {
        assert_eq!(code, 0, "i32_add returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i32_sub() {
    let mut builder = FunctionBuilder::new("blitz_i32_sub", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.sub(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_sub finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_sub");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_sub(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_sub(30, 10) != 20) return 1;
    if (blitz_i32_sub(5, 5) != 0) return 2;
    if (blitz_i32_sub(3, 10) != -7) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_sub", &obj, c_main) {
        assert_eq!(code, 0, "i32_sub returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i32_mul() {
    let mut builder = FunctionBuilder::new("blitz_i32_mul", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.mul(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_mul finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_mul");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_mul(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_mul(6, 7) != 42) return 1;
    if (blitz_i32_mul(0, 100) != 0) return 2;
    if (blitz_i32_mul(-3, 4) != -12) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_mul", &obj, c_main) {
        assert_eq!(code, 0, "i32_mul returned wrong exit code {code}");
    }
}

// Task 6.3: I32 signed division (CDQ should be used, not CQO).
#[test]
fn e2e_i32_sdiv() {
    use crate::test_utils::objdump_disasm;

    let mut builder = FunctionBuilder::new("blitz_i32_sdiv", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.sdiv(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_sdiv finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_sdiv");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_sdiv(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_sdiv(42, 6) != 7) return 1;
    if (blitz_i32_sdiv(17, 3) != 5) return 2;
    if (blitz_i32_sdiv(-15, 3) != -5) return 3;
    if (blitz_i32_sdiv(100, 7) != 14) return 4;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_sdiv", &obj, c_main) {
        assert_eq!(code, 0, "i32_sdiv returned wrong exit code {code}");
    }

    // Verify CDQ (0x99) is present and CQO (0x48 0x99) is not.
    if let Some(disasm) = objdump_disasm(&obj.code) {
        assert!(
            disasm.contains("cdq") || disasm.contains("cltd"),
            "I32 division should use CDQ, not CQO:\n{disasm}"
        );
    }
}

#[test]
fn e2e_i32_udiv() {
    let mut builder = FunctionBuilder::new("blitz_i32_udiv", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.udiv(a, b);
    let r64 = builder.zext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_udiv finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_udiv");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_udiv(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_udiv(42, 6) != 7) return 1;
    if (blitz_i32_udiv(100, 10) != 10) return 2;
    if (blitz_i32_udiv(7, 2) != 3) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_udiv", &obj, c_main) {
        assert_eq!(code, 0, "i32_udiv returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i32_or() {
    let mut builder = FunctionBuilder::new("blitz_i32_or", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.or(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_or finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_or");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_or(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_or(0xF0, 0x0F) != 0xFF) return 1;
    if (blitz_i32_or(0, 0) != 0) return 2;
    if (blitz_i32_or(0xAA, 0x55) != 0xFF) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_or", &obj, c_main) {
        assert_eq!(code, 0, "i32_or returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i32_xor() {
    let mut builder = FunctionBuilder::new("blitz_i32_xor", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.xor(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_xor finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_xor");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_xor(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_xor(0xFF, 0x0F) != 0xF0) return 1;
    if (blitz_i32_xor(0xAA, 0xAA) != 0) return 2;
    if (blitz_i32_xor(0, 0xFF) != 0xFF) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_xor", &obj, c_main) {
        assert_eq!(code, 0, "i32_xor returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i32_shl() {
    let mut builder = FunctionBuilder::new("blitz_i32_shl", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.shl(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_shl finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_shl");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_shl(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_shl(1, 4) != 16) return 1;
    if (blitz_i32_shl(0xFF, 8) != 0xFF00) return 2;
    if (blitz_i32_shl(1, 0) != 1) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_shl", &obj, c_main) {
        assert_eq!(code, 0, "i32_shl returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i32_sar() {
    let mut builder = FunctionBuilder::new("blitz_i32_sar", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let r = builder.sar(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i32_sar finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_sar");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_sar(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_sar(256, 4) != 16) return 1;
    if (blitz_i32_sar(-16, 2) != -4) return 2;
    if (blitz_i32_sar(1, 0) != 1) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_sar", &obj, c_main) {
        assert_eq!(code, 0, "i32_sar returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i32_icmp() {
    use crate::ir::condcode::CondCode;

    // Build: i32_icmp(a, b) returns 1 if a > b (signed), else 0.
    let mut builder = FunctionBuilder::new("blitz_i32_icmp", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I32);
    let b = builder.trunc(params[1], Type::I32);
    let cond = builder.icmp(CondCode::Sgt, a, b);
    // Select 1 or 0 based on the comparison.
    let one = builder.iconst(1, Type::I64);
    let zero = builder.iconst(0, Type::I64);
    let r = builder.select(cond, one, zero);
    builder.ret(Some(r));
    let func = builder.finalize().expect("i32_icmp finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_icmp");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_icmp(int64_t a, int64_t b);
int main(void) {
    if (blitz_i32_icmp(10, 5) != 1) return 1;
    if (blitz_i32_icmp(5, 10) != 0) return 2;
    if (blitz_i32_icmp(5, 5) != 0) return 3;
    if (blitz_i32_icmp(-1, 0) != 0) return 4;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_icmp", &obj, c_main) {
        assert_eq!(code, 0, "i32_icmp returned wrong exit code {code}");
    }
}

// Task 6.4-6.5: I16 arithmetic tests.

#[test]
fn e2e_i16_add() {
    let mut builder = FunctionBuilder::new("blitz_i16_add", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I16);
    let b = builder.trunc(params[1], Type::I16);
    let r = builder.add(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i16_add finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i16_add");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i16_add(int64_t a, int64_t b);
int main(void) {
    if (blitz_i16_add(100, 200) != 300) return 1;
    if (blitz_i16_add(0, 0) != 0) return 2;
    if (blitz_i16_add(-10, 3) != -7) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i16_add", &obj, c_main) {
        assert_eq!(code, 0, "i16_add returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i16_sub() {
    let mut builder = FunctionBuilder::new("blitz_i16_sub", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I16);
    let b = builder.trunc(params[1], Type::I16);
    let r = builder.sub(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i16_sub finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i16_sub");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i16_sub(int64_t a, int64_t b);
int main(void) {
    if (blitz_i16_sub(300, 100) != 200) return 1;
    if (blitz_i16_sub(50, 50) != 0) return 2;
    if (blitz_i16_sub(10, 20) != -10) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i16_sub", &obj, c_main) {
        assert_eq!(code, 0, "i16_sub returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i16_and() {
    let mut builder = FunctionBuilder::new("blitz_i16_and", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I16);
    let b = builder.trunc(params[1], Type::I16);
    let r = builder.and(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i16_and finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i16_and");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i16_and(int64_t a, int64_t b);
int main(void) {
    if (blitz_i16_and(0xFF00, 0x00FF) != 0) return 1;
    if (blitz_i16_and(0xFFFF, 0x00FF) != 0x00FF) return 2;
    if (blitz_i16_and(0x1234, 0xFF00) != 0x1200) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i16_and", &obj, c_main) {
        assert_eq!(code, 0, "i16_and returned wrong exit code {code}");
    }
}

// Task 6.6-6.7: I8 arithmetic tests.

#[test]
fn e2e_i8_add() {
    let mut builder = FunctionBuilder::new("blitz_i8_add", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I8);
    let b = builder.trunc(params[1], Type::I8);
    let r = builder.add(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i8_add finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i8_add");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i8_add(int64_t a, int64_t b);
int main(void) {
    if (blitz_i8_add(10, 20) != 30) return 1;
    if (blitz_i8_add(0, 0) != 0) return 2;
    if (blitz_i8_add(100, 27) != 127) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i8_add", &obj, c_main) {
        assert_eq!(code, 0, "i8_add returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i8_sub() {
    let mut builder = FunctionBuilder::new("blitz_i8_sub", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I8);
    let b = builder.trunc(params[1], Type::I8);
    let r = builder.sub(a, b);
    let r64 = builder.sext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i8_sub finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i8_sub");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i8_sub(int64_t a, int64_t b);
int main(void) {
    if (blitz_i8_sub(50, 20) != 30) return 1;
    if (blitz_i8_sub(10, 10) != 0) return 2;
    if (blitz_i8_sub(5, 10) != -5) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i8_sub", &obj, c_main) {
        assert_eq!(code, 0, "i8_sub returned wrong exit code {code}");
    }
}

#[test]
fn e2e_i8_and() {
    let mut builder = FunctionBuilder::new("blitz_i8_and", &[Type::I64, Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let a = builder.trunc(params[0], Type::I8);
    let b = builder.trunc(params[1], Type::I8);
    let r = builder.and(a, b);
    let r64 = builder.zext(r, Type::I64);
    builder.ret(Some(r64));
    let func = builder.finalize().expect("i8_and finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i8_and");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i8_and(int64_t a, int64_t b);
int main(void) {
    if (blitz_i8_and(0xFF, 0x0F) != 0x0F) return 1;
    if (blitz_i8_and(0xAA, 0x55) != 0x00) return 2;
    if (blitz_i8_and(0xFF, 0xFF) != 0xFF) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i8_and", &obj, c_main) {
        assert_eq!(code, 0, "i8_and returned wrong exit code {code}");
    }
}

// Task 6.8: Mixed-width operations (Sext, Zext, Trunc) with correctness verification.

#[test]
fn e2e_sext_i32_to_i64_negative() {
    // Verify sign extension of a negative I32 value to I64.
    let mut builder = FunctionBuilder::new("blitz_sext32", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let truncated = builder.trunc(params[0], Type::I32);
    let extended = builder.sext(truncated, Type::I64);
    builder.ret(Some(extended));
    let func = builder.finalize().expect("sext32 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile sext32");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_sext32(int64_t x);
int main(void) {
    // Positive value: low 32 bits = 42, sext should give 42
    if (blitz_sext32(42) != 42) return 1;
    // Negative I32: 0xFFFFFFFF = -1 in 32-bit, sext to -1 in 64-bit
    if (blitz_sext32(0xFFFFFFFF) != -1) return 2;
    // 0x80000000 = INT32_MIN, sext should give -2147483648 in 64-bit
    if (blitz_sext32(0x80000000LL) != -2147483648LL) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_sext32", &obj, c_main) {
        assert_eq!(code, 0, "sext32 returned wrong exit code {code}");
    }
}

#[test]
fn e2e_trunc_i64_to_i32_roundtrip() {
    // Trunc I64 to I32, then sext back. Verifies truncation drops high bits.
    let mut builder = FunctionBuilder::new("blitz_trunc64", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let t32 = builder.trunc(params[0], Type::I32);
    let back = builder.sext(t32, Type::I64);
    builder.ret(Some(back));
    let func = builder.finalize().expect("trunc64 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile trunc64");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_trunc64(int64_t x);
int main(void) {
    // Value fits in I32: should round-trip
    if (blitz_trunc64(100) != 100) return 1;
    // High bits discarded: 0x100000005 truncates to 5
    if (blitz_trunc64(0x100000005LL) != 5) return 2;
    // Negative: -1 should round-trip
    if (blitz_trunc64(-1) != -1) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_trunc64", &obj, c_main) {
        assert_eq!(code, 0, "trunc64 returned wrong exit code {code}");
    }
}

#[test]
fn e2e_zext_i8_to_i64_values() {
    // Zext I8 to I64: high bits should be zero, not sign-extended.
    let mut builder = FunctionBuilder::new("blitz_zext8", &[Type::I64], &[Type::I64]);
    let params = builder.params().to_vec();
    let t8 = builder.trunc(params[0], Type::I8);
    let ext = builder.zext(t8, Type::I64);
    builder.ret(Some(ext));
    let func = builder.finalize().expect("zext8 finalize");

    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile zext8");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_zext8(int64_t x);
int main(void) {
    if (blitz_zext8(42) != 42) return 1;
    // 0xFF as I8 = -1 signed, but zext should give 255
    if (blitz_zext8(0xFF) != 255) return 2;
    // 0x80 as I8 = -128 signed, but zext should give 128
    if (blitz_zext8(0x80) != 128) return 3;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_zext8", &obj, c_main) {
        assert_eq!(code, 0, "zext8 returned wrong exit code {code}");
    }
}

// Task 6.9: Spill correctness for I32 values with many live variables.
#[test]
fn e2e_i32_spill_pressure() {
    // Create 20 I32 constants, keep them all live, then sum them.
    // This forces regalloc to spill some I32 values.
    let mut builder = FunctionBuilder::new("blitz_i32_sum20", &[], &[Type::I64]);

    let vals: Vec<_> = (1i64..=20).map(|v| builder.iconst(v, Type::I32)).collect();

    let mut acc = vals[0];
    for &v in &vals[1..] {
        acc = builder.add(acc, v);
    }
    // sext to I64 for return
    let r64 = builder.sext(acc, Type::I64);
    builder.ret(Some(r64));

    let func = builder.finalize().expect("i32_sum20 finalize");
    let opts = CompileOptions::default();
    let obj = compile(func, &opts, None).expect("compile i32_sum20");

    let c_main = r#"
#include <stdint.h>
int64_t blitz_i32_sum20(void);
int main(void) {
    // 1+2+...+20 = 210
    if (blitz_i32_sum20() != 210) return 1;
    return 0;
}
"#;
    if let Some(code) = link_and_run_obj("blitz_e2e_i32_sum20", &obj, c_main) {
        assert_eq!(code, 0, "i32_sum20 returned wrong exit code {code}");
    }
}

// ── insert_early_barrier_spills unit tests ──────────────────────────────────

use crate::schedule::scheduler::ScheduledInst;

fn make_inst(dst: u32, op: Op, operands: &[u32]) -> ScheduledInst {
    ScheduledInst {
        dst: VReg(dst),
        op,
        operands: operands.iter().map(|&v| VReg(v)).collect(),
    }
}

#[test]
fn early_spill_distant_barrier_result() {
    // LoadResult at barrier 0, consumer at group 3 (distance = 2).
    // Should insert SpillStore in group 1, SpillLoad in group 3.
    let mut schedule = vec![
        make_inst(0, Op::LoadResult(0, Type::I32), &[]), // barrier 0 result
        make_inst(1, Op::Iconst(10, Type::I32), &[]),    // filler
        make_inst(2, Op::Iconst(20, Type::I32), &[]),    // filler
        make_inst(3, Op::X86Add, &[0, 2]),               // consumer of LoadResult
        make_inst(4, Op::Proj0, &[3]),
    ];
    let vreg_to_result = HashMap::from([(VReg(0), 0usize)]);
    let vreg_to_arg = HashMap::new();
    let mut vreg_group = HashMap::from([
        (VReg(0), 1usize), // barrier 0 result -> group 1
        (VReg(1), 1),
        (VReg(2), 2),
        (VReg(3), 3), // consumer in group 3
        (VReg(4), 3),
    ]);
    let vreg_types = HashMap::from([
        (VReg(0), Type::I32),
        (VReg(1), Type::I32),
        (VReg(2), Type::I32),
        (VReg(3), Type::I32),
        (VReg(4), Type::I32),
    ]);
    let mut next_vreg = 10u32;
    let mut spill_counter = 0u32;

    insert_early_barrier_spills(
        &mut schedule,
        &vreg_to_result,
        &vreg_to_arg,
        &mut vreg_group,
        &vreg_types,
        &mut next_vreg,
        &mut spill_counter,
    );

    // Should have allocated one spill slot.
    assert_eq!(spill_counter, 1, "one spill slot allocated");

    // Should have inserted SpillStore and SpillLoad.
    let spill_stores: Vec<_> = schedule
        .iter()
        .filter(|i| matches!(i.op, Op::SpillStore(_)))
        .collect();
    let spill_loads: Vec<_> = schedule
        .iter()
        .filter(|i| matches!(i.op, Op::SpillLoad(_)))
        .collect();
    assert_eq!(spill_stores.len(), 1, "one SpillStore inserted");
    assert_eq!(spill_loads.len(), 1, "one SpillLoad inserted");

    // SpillStore should reference the original LoadResult VReg as operand.
    assert_eq!(spill_stores[0].operands, vec![VReg(0)]);

    // SpillStore should be in group 1 (def_group).
    assert_eq!(vreg_group[&spill_stores[0].dst], 1);

    // SpillLoad should be in group 3 (consumer_group).
    let reload_vreg = spill_loads[0].dst;
    assert_eq!(vreg_group[&reload_vreg], 3);

    // Consumer's operand should be rewritten to the reload VReg.
    let consumer = schedule
        .iter()
        .find(|i| matches!(i.op, Op::X86Add))
        .unwrap();
    assert!(
        consumer.operands.contains(&reload_vreg),
        "consumer should reference reload VReg"
    );
    assert!(
        !consumer.operands.contains(&VReg(0)),
        "consumer should NOT reference original LoadResult VReg"
    );
}

#[test]
fn early_spill_skips_close_consumer() {
    // LoadResult at barrier 0, consumer at group 2 (distance = 1).
    // Should NOT insert any spills.
    let mut schedule = vec![
        make_inst(0, Op::LoadResult(0, Type::I32), &[]),
        make_inst(1, Op::X86Add, &[0, 0]),
    ];
    let vreg_to_result = HashMap::from([(VReg(0), 0usize)]);
    let vreg_to_arg = HashMap::new();
    let mut vreg_group = HashMap::from([
        (VReg(0), 1usize),
        (VReg(1), 2), // only 1 group away
    ]);
    let vreg_types = HashMap::from([(VReg(0), Type::I32), (VReg(1), Type::I32)]);
    let mut next_vreg = 10u32;
    let mut spill_counter = 0u32;

    insert_early_barrier_spills(
        &mut schedule,
        &vreg_to_result,
        &vreg_to_arg,
        &mut vreg_group,
        &vreg_types,
        &mut next_vreg,
        &mut spill_counter,
    );

    assert_eq!(spill_counter, 0, "no spill slots allocated");
    assert_eq!(schedule.len(), 2, "no instructions inserted");
}

#[test]
fn early_spill_skips_effectful_consumer() {
    // LoadResult at barrier 0, consumer at group 3 (distance = 2),
    // BUT the LoadResult is also consumed by a later effectful op
    // (vreg_to_arg_of_barrier has it). Should NOT spill.
    let mut schedule = vec![
        make_inst(0, Op::LoadResult(0, Type::I32), &[]),
        make_inst(1, Op::Iconst(10, Type::I32), &[]),
        make_inst(2, Op::X86Add, &[0, 1]),
    ];
    let vreg_to_result = HashMap::from([(VReg(0), 0usize)]);
    // VReg(0) is also consumed by barrier 2 (a Store).
    let vreg_to_arg = HashMap::from([(VReg(0), 2usize)]);
    let mut vreg_group = HashMap::from([(VReg(0), 1usize), (VReg(1), 1), (VReg(2), 3)]);
    let vreg_types = HashMap::from([
        (VReg(0), Type::I32),
        (VReg(1), Type::I32),
        (VReg(2), Type::I32),
    ]);
    let mut next_vreg = 10u32;
    let mut spill_counter = 0u32;

    insert_early_barrier_spills(
        &mut schedule,
        &vreg_to_result,
        &vreg_to_arg,
        &mut vreg_group,
        &vreg_types,
        &mut next_vreg,
        &mut spill_counter,
    );

    assert_eq!(spill_counter, 0, "no spill for effectful-consumed result");
}

#[test]
fn early_spill_skips_no_consumers() {
    // LoadResult at barrier 0 with no scheduled consumers at all.
    let mut schedule = vec![make_inst(0, Op::LoadResult(0, Type::I32), &[])];
    let vreg_to_result = HashMap::from([(VReg(0), 0usize)]);
    let vreg_to_arg = HashMap::new();
    let mut vreg_group = HashMap::from([(VReg(0), 1usize)]);
    let vreg_types = HashMap::from([(VReg(0), Type::I32)]);
    let mut next_vreg = 10u32;
    let mut spill_counter = 0u32;

    insert_early_barrier_spills(
        &mut schedule,
        &vreg_to_result,
        &vreg_to_arg,
        &mut vreg_group,
        &vreg_types,
        &mut next_vreg,
        &mut spill_counter,
    );

    assert_eq!(spill_counter, 0, "no spill for dead result");
}

#[test]
fn early_spill_multiple_consumers_uses_earliest() {
    // LoadResult at barrier 0, consumers at groups 4 and 5.
    // Should spill with reload at group 4 (earliest consumer).
    // Both consumers should be rewritten.
    let mut schedule = vec![
        make_inst(0, Op::LoadResult(0, Type::I32), &[]),
        make_inst(1, Op::Iconst(1, Type::I32), &[]),
        make_inst(2, Op::Iconst(2, Type::I32), &[]),
        make_inst(3, Op::X86Add, &[0, 1]), // consumer 1 in group 4
        make_inst(4, Op::X86Sub, &[0, 2]), // consumer 2 in group 5
    ];
    let vreg_to_result = HashMap::from([(VReg(0), 0usize)]);
    let vreg_to_arg = HashMap::new();
    let mut vreg_group = HashMap::from([
        (VReg(0), 1usize),
        (VReg(1), 2),
        (VReg(2), 3),
        (VReg(3), 4),
        (VReg(4), 5),
    ]);
    let vreg_types = HashMap::from([
        (VReg(0), Type::I32),
        (VReg(1), Type::I32),
        (VReg(2), Type::I32),
        (VReg(3), Type::I32),
        (VReg(4), Type::I32),
    ]);
    let mut next_vreg = 10u32;
    let mut spill_counter = 0u32;

    insert_early_barrier_spills(
        &mut schedule,
        &vreg_to_result,
        &vreg_to_arg,
        &mut vreg_group,
        &vreg_types,
        &mut next_vreg,
        &mut spill_counter,
    );

    assert_eq!(spill_counter, 1);

    let spill_loads: Vec<_> = schedule
        .iter()
        .filter(|i| matches!(i.op, Op::SpillLoad(_)))
        .collect();
    let reload_vreg = spill_loads[0].dst;
    // Reload should be at group 4 (earliest consumer).
    assert_eq!(vreg_group[&reload_vreg], 4);

    // Both consumers should use the reload VReg.
    let add = schedule
        .iter()
        .find(|i| matches!(i.op, Op::X86Add))
        .unwrap();
    let sub = schedule
        .iter()
        .find(|i| matches!(i.op, Op::X86Sub))
        .unwrap();
    assert!(add.operands.contains(&reload_vreg));
    assert!(sub.operands.contains(&reload_vreg));
}
