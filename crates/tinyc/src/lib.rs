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
    let obj = blitz::compile::compile_module(cg.functions, &opts).map_err(|e| TinyErr {
        line: 0,
        col: 0,
        msg: e.to_string(),
    })?;
    Ok(obj.finalize())
}

/// Returns the C source for the tinyc runtime helpers (div/rem support).
pub fn runtime_helpers_c() -> &'static str {
    r#"
long long __tinyc_sdiv(long long a, long long b) { return a / b; }
long long __tinyc_srem(long long a, long long b) { return a % b; }
"#
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
        let helpers_c_path = tmp_dir.join(format!("tinyc_helpers_{suffix}.c"));
        let helpers_obj_path = tmp_dir.join(format!("tinyc_helpers_{suffix}.o"));
        let bin_path = tmp_dir.join(format!("tinyc_test_bin_{suffix}"));

        // Write tinyc object file
        {
            let mut f = std::fs::File::create(&obj_path).expect("create obj file");
            f.write_all(&obj_bytes).expect("write obj bytes");
        }

        // Write and compile helpers C file
        {
            let mut f = std::fs::File::create(&helpers_c_path).expect("create helpers C file");
            f.write_all(super::runtime_helpers_c().as_bytes())
                .expect("write helpers C");
        }
        let helpers_compile = Command::new("cc")
            .arg("-c")
            .arg(&helpers_c_path)
            .arg("-o")
            .arg(&helpers_obj_path)
            .status()
            .expect("compile helpers");
        if !helpers_compile.success() {
            panic!("helpers compilation failed");
        }

        // Link with cc
        let link_status = Command::new("cc")
            .arg(&obj_path)
            .arg(&helpers_obj_path)
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
        let _ = std::fs::remove_file(&helpers_c_path);
        let _ = std::fs::remove_file(&helpers_obj_path);
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
}
