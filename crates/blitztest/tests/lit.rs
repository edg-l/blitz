use std::path::PathBuf;
use std::process::Command;

fn find_workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = std::fs::read_to_string(&cargo_toml).unwrap();
            if content.contains("[workspace]") {
                return dir;
            }
        }
        if !dir.pop() {
            panic!("could not find workspace root");
        }
    }
}

#[test]
fn run_lit_tests() {
    let root = find_workspace_root();

    // Build tinyc first
    let build = Command::new("cargo")
        .arg("build")
        .arg("-p")
        .arg("tinyc")
        .current_dir(&root)
        .status()
        .expect("cargo build failed");
    assert!(build.success(), "cargo build -p tinyc failed");

    let tinyc_path = root.join("target/debug/tinyc");
    let test_dir = root.join("tests/lit");

    let config = blitztest::runner::Config {
        tinyc_path,
        test_dir: test_dir.clone(),
    };

    let tests = blitztest::discover::discover_tests(&test_dir);
    assert!(
        !tests.is_empty(),
        "no lit tests found in {}",
        test_dir.display()
    );

    let mut failures = Vec::new();
    for test_path in &tests {
        let result = blitztest::runner::run_test(test_path, &config);
        match result.status {
            blitztest::runner::TestStatus::Pass => {}
            blitztest::runner::TestStatus::Fail(msg) => {
                failures.push((result.name, msg));
            }
            blitztest::runner::TestStatus::Skip(_) => {}
        }
    }

    if !failures.is_empty() {
        let mut msg = format!("{} lit test(s) failed:\n", failures.len());
        for (name, reason) in &failures {
            msg.push_str(&format!("  FAIL: {name}: {reason}\n"));
        }
        panic!("{msg}");
    }
}
