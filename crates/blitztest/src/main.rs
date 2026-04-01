use std::path::PathBuf;
use std::process::exit;

use blitztest::discover::discover_tests;
use blitztest::runner::{Config, TestStatus, run_test};

fn find_workspace_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists()
            && std::fs::read_to_string(&cargo_toml)
                .map(|content| content.contains("[workspace]"))
                .unwrap_or(false)
        {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn find_tinyc() -> Option<PathBuf> {
    // Check TINYC env var first.
    if let Ok(path) = std::env::var("TINYC") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    // Check target/debug/tinyc relative to workspace root.
    if let Some(root) = find_workspace_root() {
        let candidate = root.join("target/debug/tinyc");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Try which tinyc.
    if let Some(output) = std::process::Command::new("which")
        .arg("tinyc")
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return (!path.is_empty()).then(|| PathBuf::from(path));
    }

    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut test_dir = None;
    let mut tinyc_path = None;
    let mut filter = None;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--test-dir" => {
                if i + 1 < args.len() {
                    test_dir = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("--test-dir requires an argument");
                    exit(1);
                }
            }
            "--tinyc" => {
                if i + 1 < args.len() {
                    tinyc_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("--tinyc requires an argument");
                    exit(1);
                }
            }
            "--filter" => {
                if i + 1 < args.len() {
                    filter = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--filter requires an argument");
                    exit(1);
                }
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let test_dir = test_dir.unwrap_or_else(|| {
        find_workspace_root()
            .map(|r| r.join("tests/lit"))
            .unwrap_or_else(|| PathBuf::from("tests/lit"))
    });

    let tinyc_path = tinyc_path.unwrap_or_else(|| {
        find_tinyc().unwrap_or_else(|| {
            eprintln!("error: cannot find tinyc binary. Use --tinyc or set TINYC env var.");
            exit(1);
        })
    });

    let config = Config {
        tinyc_path,
        test_dir: test_dir.clone(),
    };

    let mut tests = discover_tests(&test_dir);
    if let Some(ref pattern) = filter {
        tests.retain(|p| p.to_string_lossy().contains(pattern.as_str()));
    }

    if tests.is_empty() {
        eprintln!("no tests found in {}", test_dir.display());
        exit(1);
    }

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();

    for test_path in &tests {
        let result = run_test(test_path, &config);
        match &result.status {
            TestStatus::Pass => {
                passed += 1;
                if verbose {
                    println!("PASS: {}", result.name);
                } else {
                    print!(".");
                }
            }
            TestStatus::Fail(msg) => {
                failed += 1;
                if verbose {
                    println!("FAIL: {}: {}", result.name, msg);
                } else {
                    print!("F");
                }
                failures.push((result.name.clone(), msg.clone()));
            }
            TestStatus::Skip(msg) => {
                skipped += 1;
                if verbose {
                    println!("SKIP: {}: {}", result.name, msg);
                } else {
                    print!("S");
                }
            }
        }
    }

    if !verbose {
        println!();
    }

    let total = passed + failed + skipped;
    println!("\n{total} tests: {passed} passed, {failed} failed, {skipped} skipped");

    if !failures.is_empty() {
        println!("\nFailures:");
        for (name, reason) in &failures {
            println!("  FAIL: {name}: {reason}");
        }
        exit(1);
    }
}
