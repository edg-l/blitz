use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::check::run_checks;
use crate::directive::{CheckPattern, Directive, parse_directives};

pub struct Config {
    pub tinyc_path: PathBuf,
    pub test_dir: PathBuf,
}

pub enum TestStatus {
    Pass,
    Fail(String),
    Skip(String),
}

pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
}

/// Run a single lit test file.
pub fn run_test(test_path: &Path, config: &Config) -> TestResult {
    let name = test_path
        .strip_prefix(&config.test_dir)
        .unwrap_or(test_path)
        .display()
        .to_string();

    let source = match std::fs::read_to_string(test_path) {
        Ok(s) => s,
        Err(e) => {
            return TestResult {
                name,
                status: TestStatus::Fail(format!("cannot read file: {e}")),
            };
        }
    };

    let directives = match parse_directives(&source) {
        Ok(d) => d,
        Err(e) => {
            return TestResult {
                name,
                status: TestStatus::Fail(format!("directive parse error: {e}")),
            };
        }
    };

    // Collect RUN, CHECK, and EXIT directives.
    let run_cmds: Vec<&str> = directives
        .iter()
        .filter_map(|d| match d {
            Directive::Run { cmd, .. } => Some(cmd.as_str()),
            _ => None,
        })
        .collect();

    if run_cmds.is_empty() {
        return TestResult {
            name,
            status: TestStatus::Skip("no RUN directives".into()),
        };
    }

    let checks: Vec<&CheckPattern> = directives
        .iter()
        .filter_map(|d| match d {
            Directive::Check(p) => Some(p),
            _ => None,
        })
        .collect();

    let exit_directive: Option<i32> = directives.iter().find_map(|d| match d {
        Directive::Exit { code, .. } => Some(*code),
        _ => None,
    });

    // Generate unique temp path from test file path.
    let canonical = test_path
        .canonicalize()
        .unwrap_or_else(|_| test_path.to_path_buf());
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    let hash = hasher.finish();
    let temp_base = std::env::temp_dir().join(format!("blitztest_{hash:x}"));

    let test_path_str = test_path.to_string_lossy();
    let tinyc_str = config.tinyc_path.to_string_lossy();
    let temp_str = temp_base.to_string_lossy();

    // Execute RUN lines.
    let mut last_output = String::new();
    let mut last_exit_code: i32 = 0;

    for cmd_template in &run_cmds {
        let cmd = cmd_template
            .replace("%tinyc", &tinyc_str)
            .replace("%s", &test_path_str)
            .replace("%t", &temp_str);

        let result = Command::new("sh").arg("-c").arg(&cmd).output();

        match result {
            Ok(output) => {
                last_exit_code = output.status.code().unwrap_or(-1);
                last_output = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.is_empty() {
                    last_output.push_str(&stderr);
                }

                // If command failed and no EXIT directive, it's a test failure.
                if !output.status.success() && exit_directive.is_none() {
                    cleanup_temp(&temp_base);
                    return TestResult {
                        name,
                        status: TestStatus::Fail(format!(
                            "command failed (exit {}): {cmd}\n{last_output}",
                            last_exit_code
                        )),
                    };
                }
            }
            Err(e) => {
                cleanup_temp(&temp_base);
                return TestResult {
                    name,
                    status: TestStatus::Fail(format!("failed to execute: {cmd}: {e}")),
                };
            }
        }
    }

    // Check EXIT directive against last command's exit code.
    if let Some(expected_code) = exit_directive {
        // Normalize: process exit codes are typically 0-255 on Unix.
        // When a process calls exit(42), the shell reports status 42.
        // But for values > 255, take modulo 256 to match shell behavior.
        let actual = last_exit_code;
        // On Unix, exit status is the low 8 bits.
        let expected_normalized = expected_code & 0xFF;
        let actual_normalized = actual & 0xFF;
        if actual_normalized != expected_normalized {
            cleanup_temp(&temp_base);
            return TestResult {
                name,
                status: TestStatus::Fail(format!(
                    "expected exit code {expected_code}, got {actual}"
                )),
            };
        }
    }

    // Run CHECK directives against last output.
    if !checks.is_empty() {
        let check_patterns: Vec<&CheckPattern> = checks;
        let borrowed: Vec<CheckPattern> = check_patterns
            .iter()
            .map(|p| CheckPattern {
                kind: p.kind.clone(),
                raw: p.raw.clone(),
                regex: p.regex.clone(),
                line_no: p.line_no,
            })
            .collect();
        if let Err(e) = run_checks(&last_output, &borrowed) {
            cleanup_temp(&temp_base);
            return TestResult {
                name,
                status: TestStatus::Fail(format!("{e}")),
            };
        }
    }

    cleanup_temp(&temp_base);
    TestResult {
        name,
        status: TestStatus::Pass,
    }
}

/// Clean up temp files: remove %t and any %t.* files.
fn cleanup_temp(temp_base: &Path) {
    let _ = std::fs::remove_file(temp_base);
    // Also try to remove common extensions
    if let (Some(parent), Some(stem)) = (temp_base.parent(), temp_base.file_name()) {
        let prefix = stem.to_string_lossy();
        {
            if let Ok(entries) = std::fs::read_dir(parent) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with(&*prefix) {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }
    }
}
