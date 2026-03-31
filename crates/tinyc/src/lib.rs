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

        // Use a unique suffix per test (thread id) to avoid conflicts
        let tid = std::thread::current().id();
        let suffix = format!("{tid:?}").replace(['(', ')'], "").replace(' ', "_");
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
        let src = "int main() { long x = 255; long y = 15; return (int)(x & y); }";
        assert_eq!(compile_and_run(src), Some(15));
    }

    #[test]
    fn test_bitwise_or() {
        let src = "int main() { long x = 240; long y = 15; return (int)(x | y); }";
        assert_eq!(compile_and_run(src), Some(255));
    }

    #[test]
    fn test_bitwise_xor() {
        let src = "int main() { long x = 255; long y = 15; return (int)(x ^ y); }";
        assert_eq!(compile_and_run(src), Some(240));
    }

    #[test]
    fn test_shift_left() {
        let src = "int main() { long x = 1; long y = 4; return (int)(x << y); }";
        assert_eq!(compile_and_run(src), Some(16));
    }

    #[test]
    fn test_shift_right() {
        let src = "int main() { long x = 256; long y = 4; return (int)(x >> y); }";
        assert_eq!(compile_and_run(src), Some(16));
    }

    #[test]
    fn test_bitwise_not() {
        let src = "int main() { long x = 255; long mask = 255; return (int)(~x & mask); }";
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
}
