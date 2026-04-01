use std::path::{Path, PathBuf};

/// Recursively find all `.c` files under `dir`, sorted by path.
pub fn discover_tests(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    discover_recursive(dir, &mut results);
    results.sort();
    results
}

fn discover_recursive(dir: &Path, results: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            discover_recursive(&path, results);
        } else if path.extension().is_some_and(|ext| ext == "c") {
            results.push(path);
        }
    }
}
