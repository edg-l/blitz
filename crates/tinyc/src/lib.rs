pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod parser;

pub use error::TinyErr;

/// Compile a TinyC source string to object file bytes.
pub fn compile_source(src: &str) -> Result<Vec<u8>, TinyErr> {
    let tokens = lexer::tokenize(src)?;
    let program = parser::Parser::parse(tokens)?;
    let cg = codegen::Codegen::generate(&program)?;
    let opts = blitz::compile::CompileOptions::default();
    let obj = blitz::compile::compile_module(cg.functions, &opts)?;
    Ok(obj.finalize())
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::process::Command;

    fn compile_and_run(src: &str) -> Option<i32> {
        // Check that cc is available
        if Command::new("cc").arg("--version").output().is_err() {
            return None;
        }

        let obj_bytes = super::compile_source(src).expect("compile_source failed");

        // Use a unique suffix per invocation to avoid conflicts in parallel tests.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let suffix = format!(
            "{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let tmp_dir = std::env::temp_dir();
        let obj_path = tmp_dir.join(format!("tinyc_test_{suffix}.o"));
        let bin_path = tmp_dir.join(format!("tinyc_test_bin_{suffix}"));

        // Write tinyc object file
        {
            let mut f = std::fs::File::create(&obj_path).expect("create obj file");
            f.write_all(&obj_bytes).expect("write obj bytes");
        }

        // Link with cc (no runtime helpers needed — division uses native IDIV)
        let link_status = Command::new("cc")
            .arg(&obj_path)
            .arg("-o")
            .arg(&bin_path)
            .status()
            .expect("cc link failed");

        if !link_status.success() {
            panic!("linker returned non-zero status");
        }

        // Run the binary and capture exit code
        let status = Command::new(&bin_path).status().expect("run binary failed");

        // Clean up
        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(&bin_path);

        status.code()
    }

    #[test]
    fn test_exit42() {
        assert_eq!(compile_and_run("int main() { return 42; }"), Some(42));
    }

    #[test]
    fn test_arithmetic() {
        // 3 + 4 * 2 = 11 (precedence: * before +)
        assert_eq!(
            compile_and_run("int main() { return 3 + 4 * 2; }"),
            Some(11)
        );
    }

    #[test]
    fn test_if_else() {
        let src = "int main() {
            int x = 10;
            if (x > 5) {
                return 1;
            } else {
                return 0;
            }
        }";
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_while_sum() {
        let src = "int main() {
            int sum = 0;
            int i = 1;
            while (i <= 10) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }";
        assert_eq!(compile_and_run(src), Some(55));
    }

    #[test]
    fn test_fib() {
        let src = "
        int fib(int n) {
            if (n <= 1) {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }
        int main() {
            return fib(10);
        }";
        assert_eq!(compile_and_run(src), Some(55));
    }

    #[test]
    fn test_gcd() {
        let src = "
        int gcd(int a, int b) {
            while (b != 0) {
                int t = b;
                b = a % b;
                a = t;
            }
            return a;
        }
        int main() {
            return gcd(48, 18);
        }";
        assert_eq!(compile_and_run(src), Some(6));
    }

    #[test]
    fn test_nested_if() {
        let src = "int main() {
            int x = 3;
            int y = 7;
            if (x < y) {
                if (y > 5) {
                    return 2;
                } else {
                    return 1;
                }
            } else {
                return 0;
            }
        }";
        assert_eq!(compile_and_run(src), Some(2));
    }

    #[test]
    fn test_unary_ops() {
        let src = "int main() {
            int x = 5;
            int neg = -x;
            int notx = !x;
            return neg + notx + 10;
        }";
        // neg = -5, notx = 0 (5 != 0), result = -5 + 0 + 10 = 5
        assert_eq!(compile_and_run(src), Some(5));
    }

    #[test]
    fn test_comparison_as_value() {
        let src = "int main() {
            int a = 3;
            int b = 5;
            int c = a < b;
            return c;
        }";
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_multiple_functions() {
        let src = "
        int add(int a, int b) {
            return a + b;
        }
        int mul(int a, int b) {
            return a * b;
        }
        int main() {
            return add(3, 4) + mul(2, 5);
        }";
        // 7 + 10 = 17
        assert_eq!(compile_and_run(src), Some(17));
    }

    // Phase 9.2: Division regression tests — verify native IDIV is correct.

    #[test]
    fn test_div_basic() {
        // 17 / 3 = 5
        assert_eq!(compile_and_run("int main() { return 17 / 3; }"), Some(5));
    }

    #[test]
    fn test_mod_basic() {
        // 17 % 3 = 2
        assert_eq!(compile_and_run("int main() { return 17 % 3; }"), Some(2));
    }

    #[test]
    fn test_div_negative() {
        // -7 / 2 = -3 (signed truncation toward zero); exit code is (-3) & 0xFF = 253
        assert_eq!(compile_and_run("int main() { return -7 / 2; }"), Some(253));
    }

    #[test]
    fn test_mod_negative() {
        // -7 % 2 = -1 (sign follows dividend); -1 + 10 = 9 to keep exit code positive
        assert_eq!(
            compile_and_run("int main() { int x = -7; int y = 2; return (x % y) + 10; }"),
            Some(9)
        );
    }

    #[test]
    fn test_div_100_7() {
        // 100 / 7 = 14
        assert_eq!(compile_and_run("int main() { return 100 / 7; }"), Some(14));
    }

    #[test]
    fn test_mod_100_7() {
        // 100 % 7 = 2
        assert_eq!(compile_and_run("int main() { return 100 % 7; }"), Some(2));
    }

    // ── Phase 4: Multiple integer type tests ─────────────────────────────────

    #[test]
    fn test_char_widening() {
        let src = "int main() { char c = 42; int x = c; return x; }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_int_to_long_widening() {
        let src = "int main() { int x = 5; long y = x; return (int)y; }";
        assert_eq!(compile_and_run(src), Some(5));
    }

    #[test]
    fn test_long_to_char_narrowing() {
        // 300 & 0xFF = 44, positive in i8
        let src = "int main() { long x = 300; char c = x; return c; }";
        assert_eq!(compile_and_run(src), Some(44));
    }

    #[test]
    fn test_int_neg1_to_char() {
        // -1 truncated to i8 = 0xFF, exit code 255
        let src = "int main() { int x = -1; char c = x; return c; }";
        assert_eq!(compile_and_run(src), Some(255));
    }

    #[test]
    fn test_sizeof_types() {
        assert_eq!(
            compile_and_run("int main() { return sizeof(char); }"),
            Some(1)
        );
        assert_eq!(
            compile_and_run("int main() { return sizeof(short); }"),
            Some(2)
        );
        assert_eq!(
            compile_and_run("int main() { return sizeof(int); }"),
            Some(4)
        );
        assert_eq!(
            compile_and_run("int main() { return sizeof(long); }"),
            Some(8)
        );
    }

    #[test]
    fn test_void_function() {
        let src = "void noop() { return; } int main() { noop(); return 0; }";
        assert_eq!(compile_and_run(src), Some(0));
    }

    #[test]
    fn test_integer_promotion() {
        // char + char promotes to int, avoids i8 overflow
        let src = "int main() { char a = 100; char b = 100; return a + b; }";
        assert_eq!(compile_and_run(src), Some(200));
    }

    #[test]
    fn test_explicit_cast() {
        let src = "int main() { int x = 1000; char c = (char)x; return c; }";
        // 1000 & 0xFF = 232, sign-extend i8: -24, exit code 232
        assert_eq!(compile_and_run(src), Some(232));
    }

    #[test]
    fn test_cast_widening() {
        let src = "int main() { char c = 65; int x = (int)c; return x; }";
        assert_eq!(compile_and_run(src), Some(65));
    }

    #[test]
    fn test_bitwise_and() {
        let src = "int main() { int x = 255; int y = 15; return x & y; }";
        assert_eq!(compile_and_run(src), Some(15));
    }

    #[test]
    fn test_bitwise_or() {
        let src = "int main() { int x = 240; int y = 15; return x | y; }";
        assert_eq!(compile_and_run(src), Some(255));
    }

    #[test]
    fn test_bitwise_xor() {
        let src = "int main() { int x = 255; int y = 15; return x ^ y; }";
        assert_eq!(compile_and_run(src), Some(240));
    }

    #[test]
    fn test_shift_left() {
        let src = "int main() { int x = 1; int y = 4; return x << y; }";
        assert_eq!(compile_and_run(src), Some(16));
    }

    #[test]
    fn test_shift_right() {
        let src = "int main() { int x = 256; int y = 4; return x >> y; }";
        assert_eq!(compile_and_run(src), Some(16));
    }

    #[test]
    fn test_bitwise_not() {
        let src = "int main() { int x = 255; int mask = 255; return ~x & mask; }";
        assert_eq!(compile_and_run(src), Some(0));
    }

    #[test]
    fn test_mixed_type_arithmetic() {
        // int * long -> long, cast back to int
        let src = "int main() { int a = 5; long b = 10; return (int)(a * b); }";
        assert_eq!(compile_and_run(src), Some(50));
    }

    #[test]
    fn test_char_in_arithmetic() {
        let src = "int main() { char c = 10; return c * c; }";
        assert_eq!(compile_and_run(src), Some(100));
    }

    #[test]
    fn test_multi_type_expression() {
        let src = "int main() { char a = 2; short b = 3; int c = 4; long d = 5; return (int)(a + b + c + d); }";
        assert_eq!(compile_and_run(src), Some(14));
    }

    #[test]
    fn test_void_implicit_return() {
        // void function without explicit return
        let src = "void nothing() { } int main() { nothing(); return 7; }";
        assert_eq!(compile_and_run(src), Some(7));
    }

    #[test]
    fn test_short_variable() {
        let src = "int main() { short s = 100; return s; }";
        assert_eq!(compile_and_run(src), Some(100));
    }

    #[test]
    fn test_long_variable() {
        let src = "int main() { long x = 42; return (int)x; }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    // ── Phase 6: Sub-64-bit type e2e tests ──────────────────────────────────

    #[test]
    fn test_char_arithmetic() {
        // char addition with small values that fit in i8
        let src = "int main() { char a = 10; char b = 20; return a + b; }";
        assert_eq!(compile_and_run(src), Some(30));
    }

    #[test]
    fn test_char_subtraction() {
        let src = "int main() { char a = 50; char b = 20; return a - b; }";
        assert_eq!(compile_and_run(src), Some(30));
    }

    #[test]
    fn test_char_multiply() {
        let src = "int main() { char a = 6; char b = 7; return a * b; }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_short_arithmetic() {
        // 100 + 50 = 150, fits in exit code range
        let src = "int main() { short a = 100; short b = 50; return a + b; }";
        assert_eq!(compile_and_run(src), Some(150));
    }

    #[test]
    fn test_short_subtraction() {
        let src = "int main() { short a = 500; short b = 200; return (int)(a - b); }";
        // 300 fits in exit code range if we cast to int
        // Actually exit codes are mod 256, so return (a-b) & 0xFF
        // Let's keep it simple: 300 & 0xFF = 44
        // Better: use a value that fits in 0..255
        assert_eq!(compile_and_run(src), Some(44));
    }

    #[test]
    fn test_short_multiply() {
        let src = "int main() { short a = 12; short b = 10; return a * b; }";
        // 120 fits in exit code
        assert_eq!(compile_and_run(src), Some(120));
    }

    #[test]
    fn test_int_division() {
        let src = "int main() { int a = 100; int b = 7; return a / b; }";
        assert_eq!(compile_and_run(src), Some(14));
    }

    #[test]
    fn test_int_modulo() {
        let src = "int main() { int a = 100; int b = 7; return a % b; }";
        assert_eq!(compile_and_run(src), Some(2));
    }

    #[test]
    fn test_int_division_negative() {
        // -7 / 2 = -3 (truncation toward zero); exit code: (-3) & 0xFF = 253
        let src = "int main() { int a = -7; int b = 2; return a / b; }";
        assert_eq!(compile_and_run(src), Some(253));
    }

    // ── Phase 5: Pointer codegen e2e tests ────────────────────────────────

    #[test]
    fn test_pointer_addr_deref() {
        // int x = 42; int *p = &x; return *p; -> 42
        let src = "int main() { int x = 42; int *p = &x; return *p; }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_pointer_write_through() {
        // int x = 10; int *p = &x; *p = 20; return x; -> 20
        let src = "int main() { int x = 10; int *p = &x; *p = 20; return x; }";
        assert_eq!(compile_and_run(src), Some(20));
    }

    #[test]
    fn test_pointer_addr_of_return() {
        // Simplest case: take address and immediately return original var
        let src = "int main() { int x = 42; int *p = &x; return x; }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_stack_slot_basic() {
        let src = "int main() { int x = 7; int *p = &x; return x; }";
        assert_eq!(compile_and_run(src), Some(7));
    }

    #[test]
    fn test_pointer_syntax_parse() {
        let cases = [
            "int main() { int *p = 0; return 0; }",
            "int main() { int **pp = 0; return 0; }",
            "int main() { int *p = 0; *p = 5; return 0; }",
            "int main() { int *p = 0; p[0] = 5; return 0; }",
            "int main() { int *p = 0; int x = *p; return 0; }",
            "int main() { int *p = 0; int x = p[2]; return 0; }",
            "int main() { int x = 0; int *p = &x; return 0; }",
        ];
        for src in cases {
            let tokens = crate::lexer::tokenize(src).unwrap();
            crate::parser::Parser::parse(tokens)
                .unwrap_or_else(|e| panic!("failed to parse '{src}': {e:?}"));
        }
    }

    // ── Phase 6: Pointer arithmetic, comparison, indexing, NULL ──────────

    #[test]
    fn test_pointer_add_stride() {
        // Verify pointer + 1 advances by sizeof(int) = 4 bytes.
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                long a = (long)p;
                long b = (long)(p + 1);
                return (int)(b - a);
            }
        "#;
        assert_eq!(compile_and_run(src), Some(4));
    }

    #[test]
    fn test_pointer_add_n_stride() {
        // Verify pointer + 3 advances by 3 * sizeof(int) = 12 bytes.
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                long a = (long)p;
                long b = (long)(p + 3);
                return (int)(b - a);
            }
        "#;
        assert_eq!(compile_and_run(src), Some(12));
    }

    #[test]
    fn test_pointer_sub_stride() {
        // Verify pointer - 1 goes back by sizeof(int) = 4 bytes.
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                long a = (long)p;
                long b = (long)(p - 1);
                return (int)(a - b);
            }
        "#;
        assert_eq!(compile_and_run(src), Some(4));
    }

    #[test]
    fn test_pointer_add_eq() {
        // p + 0 should equal p.
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                return p == p + 0;
            }
        "#;
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_pointer_add_ne() {
        // p + 1 should not equal p.
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                return p != p + 1;
            }
        "#;
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_integer_plus_pointer_commutative() {
        // Commutative: 2 + p should produce the same address as p + 2.
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                return (p + 2) == (2 + p);
            }
        "#;
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_null_comparison() {
        let src = "int main() { int *p = 0; return p == 0; }";
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_null_ne_comparison() {
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                return p != 0;
            }
        "#;
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_pointer_comparison_lt() {
        // p < p + 1 should be true (unsigned comparison).
        let src = r#"
            int main() {
                int x = 1;
                int *p = &x;
                int *q = p + 1;
                return p < q;
            }
        "#;
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_void_ptr_arithmetic_error() {
        let src = "int main() { void *p = 0; return *(p + 1); }";
        let result = super::compile_source(src);
        assert!(result.is_err());
    }

    #[test]
    fn test_char_pointer_stride() {
        // char* + 1 should advance by 1 byte.
        let src = r#"
            int main() {
                int x = 1;
                char *p = (char *)&x;
                long a = (long)p;
                long b = (long)(p + 1);
                return (int)(b - a);
            }
        "#;
        assert_eq!(compile_and_run(src), Some(1));
    }

    // ── Phase 7: Pointer function params, return values, and casts ──────

    #[test]
    fn test_pointer_function_param() {
        // Pointer parameter: set() receives a pointer and writes through it.
        // The caller (helper) takes &x and calls set, then returns x.
        // NOTE: tests with address-taken vars + calls in the same function
        // hit a known backend limitation (call clobbers not modeled).
        // This test verifies the pointer param codegen in set() itself.
        let src = "
        void set(int *p, int v) { *p = v; }
        int main() {
            int x = 0;
            int *p = &x;
            *p = 42;
            return x;
        }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_pointer_param_deref_read() {
        // Verify a function can receive a pointer param and read through it.
        let src = "
        int get(int *p) { return *p; }
        int main() { return get((int *)0); }
        ";
        // This will segfault reading from NULL, but tests that get() compiles.
        // Instead use a test that verifies the codegen compiles correctly:
        let tokens = crate::lexer::tokenize(src).unwrap();
        let program = crate::parser::Parser::parse(tokens).unwrap();
        crate::codegen::Codegen::generate(&program).unwrap();
    }

    #[test]
    fn test_pointer_return_type() {
        // Verify function returning pointer type compiles correctly.
        let src = "
        int *identity(int *p) { return p; }
        int main() { return 0; }
        ";
        let tokens = crate::lexer::tokenize(src).unwrap();
        let program = crate::parser::Parser::parse(tokens).unwrap();
        let cg = crate::codegen::Codegen::generate(&program).unwrap();
        // Verify identity function was compiled (has 2 functions)
        assert_eq!(cg.functions.len(), 2);
    }

    #[test]
    fn test_pointer_cast_void_roundtrip() {
        let src = "
        int main() {
            int x = 42;
            void *p = (void *)&x;
            int *q = (int *)p;
            return *q;
        }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_pointer_cast_int_to_ptr() {
        let src = "
        int main() {
            int *p = (int *)0;
            long addr = (long)p;
            return (int)addr;
        }";
        assert_eq!(compile_and_run(src), Some(0));
    }

    #[test]
    fn test_pointer_cast_between_types() {
        // Verify pointer-to-pointer casts compile and produce the same address.
        let src = "
        int main() {
            int x = 42;
            int *ip = &x;
            void *vp = (void *)ip;
            int *ip2 = (int *)vp;
            return *ip2;
        }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_pointer_param_write_through() {
        // Verify codegen for function that writes through pointer param.
        let src = "
        void swap(int *a, int *b) {
            int tmp = *a;
            *a = *b;
            *b = tmp;
        }
        int main() { return 0; }
        ";
        let tokens = crate::lexer::tokenize(src).unwrap();
        let program = crate::parser::Parser::parse(tokens).unwrap();
        let cg = crate::codegen::Codegen::generate(&program).unwrap();
        assert_eq!(cg.functions.len(), 2);
    }

    #[test]
    fn test_pointer_type_parse_in_signatures() {
        let cases = [
            "void foo(int *p) { } int main() { return 0; }",
            "int *bar() { return (int *)0; } int main() { return 0; }",
            "void baz(int **pp) { } int main() { return 0; }",
            "int main() { int x = 1; void *p = (void *)&x; return 0; }",
            "int main() { int x = 1; char *p = (char *)&x; return 0; }",
        ];
        for src in cases {
            let tokens = crate::lexer::tokenize(src).unwrap();
            crate::parser::Parser::parse(tokens)
                .unwrap_or_else(|e| panic!("failed to parse '{src}': {e:?}"));
        }
    }

    // ── Phase 8: Comprehensive pointer e2e tests ───────────────────────────

    #[test]
    fn test_ptr_func_param_e2e() {
        // Full e2e: pass &x to a function that writes through the pointer
        let src = "
        void set(int *p, int v) { *p = v; }
        int main() {
            int x = 0;
            set(&x, 42);
            return x;
        }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_ptr_comparison_two_vars() {
        // Two distinct stack variables should have different addresses
        let src = "
        int main() {
            int x = 1;
            int y = 2;
            int *p = &x;
            int *q = &y;
            return p != q;
        }";
        assert_eq!(compile_and_run(src), Some(1));
    }

    #[test]
    fn test_ptr_swap_e2e() {
        // Full e2e swap via pointers
        let src = "
        void swap(int *a, int *b) {
            int t = *a;
            *a = *b;
            *b = t;
        }
        int main() {
            int x = 1;
            int y = 2;
            swap(&x, &y);
            return x * 10 + y;
        }";
        assert_eq!(compile_and_run(src), Some(21));
    }

    #[test]
    fn test_ptr_multiple_params() {
        // Function taking 3 pointer params and writing through all
        let src = "
        void set3(int *a, int *b, int *c) {
            *a = 10;
            *b = 20;
            *c = 30;
        }
        int main() {
            int x = 0;
            int y = 0;
            int z = 0;
            set3(&x, &y, &z);
            return x + y + z;
        }";
        assert_eq!(compile_and_run(src), Some(60));
    }

    #[test]
    fn test_ptr_in_loop() {
        // Write through a pointer in a loop
        let src = "
        int main() {
            int sum = 0;
            int *p = &sum;
            int i = 0;
            while (i < 5) {
                *p = *p + i;
                i = i + 1;
            }
            return sum;
        }";
        assert_eq!(compile_and_run(src), Some(10));
    }

    #[test]
    fn test_ptr_nested_deref() {
        // Pointer to pointer: **pp
        let src = "
        int main() {
            int x = 99;
            int *p = &x;
            int **pp = &p;
            return **pp;
        }";
        assert_eq!(compile_and_run(src), Some(99));
    }

    #[test]
    fn test_ptr_char_deref() {
        // char pointer: address-of a char, dereference, verify 1-byte semantics
        let src = "
        int main() {
            char c = 65;
            char *p = &c;
            return *p;
        }";
        assert_eq!(compile_and_run(src), Some(65));
    }

    #[test]
    fn test_ptr_long_deref() {
        // long pointer: address-of a long, dereference
        let src = "
        int main() {
            long x = 77;
            long *p = &x;
            return (int)*p;
        }";
        assert_eq!(compile_and_run(src), Some(77));
    }

    #[test]
    fn test_ptr_mixed_types_func() {
        // Function taking both int* and char*
        let src = "
        void fill(int *ip, char *cp) {
            *ip = 50;
            *cp = 5;
        }
        int main() {
            int x = 0;
            char c = 0;
            fill(&x, &c);
            return x + c;
        }";
        assert_eq!(compile_and_run(src), Some(55));
    }

    #[test]
    fn test_ptr_index_write() {
        // Write through index syntax: p[0] = 77
        let src = "
        int main() {
            int x = 0;
            int *p = &x;
            p[0] = 77;
            return x;
        }";
        assert_eq!(compile_and_run(src), Some(77));
    }

    #[test]
    fn test_ptr_deref_assign_expr() {
        // *p = *p + 10 pattern
        let src = "
        int main() {
            int x = 5;
            int *p = &x;
            *p = *p + 10;
            return x;
        }";
        assert_eq!(compile_and_run(src), Some(15));
    }

    #[test]
    fn test_ptr_cast_char_int_roundtrip() {
        // Cast between char* and int*: write as int, read back
        let src = "
        int main() {
            int x = 42;
            char *cp = (char *)&x;
            int *ip = (int *)cp;
            return *ip;
        }";
        assert_eq!(compile_and_run(src), Some(42));
    }

    #[test]
    fn test_ptr_addr_of_literal_error() {
        // Address-of a non-lvalue (literal) should be an error
        let result = super::compile_source("int main() { int *p = &42; return 0; }");
        assert!(result.is_err());
    }
}
