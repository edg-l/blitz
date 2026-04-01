//! Linker abstraction: tries `ld` directly, falls back to `cc`.
//!
//! Direct `ld` invocation avoids requiring a full C compiler toolchain.
//! CRT object paths and dynamic linker are auto-detected from standard
//! locations on Linux x86-64.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Link an object file into an executable.
///
/// Tries direct `ld` first (finding CRT objects automatically).
/// Falls back to `cc` if CRT objects are not found or `ld` fails.
pub fn link(obj_path: &Path, output_path: &Path) -> Result<(), String> {
    if let Some(crt) = find_crt_paths() {
        let status = Command::new("ld")
            .arg("-dynamic-linker")
            .arg(&crt.dynamic_linker)
            .arg(&crt.crt1)
            .arg(&crt.crti)
            .arg(obj_path)
            .arg("-lc")
            .arg(&crt.crtn)
            .arg("-o")
            .arg(output_path)
            .status();

        match status {
            Ok(s) if s.success() => return Ok(()),
            Ok(s) => {
                // ld found but failed; fall through to cc
                eprintln!(
                    "tinyc: ld returned {} -- falling back to cc",
                    s.code().unwrap_or(-1)
                );
            }
            Err(_) => {
                // ld not found; fall through to cc
            }
        }
    }

    link_with_cc(obj_path, output_path)
}

fn link_with_cc(obj_path: &Path, output_path: &Path) -> Result<(), String> {
    let status = Command::new("cc")
        .arg(obj_path)
        .arg("-o")
        .arg(output_path)
        .status()
        .map_err(|e| format!("cannot run cc: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err("linker failed".to_string())
    }
}

struct CrtPaths {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    dynamic_linker: PathBuf,
}

fn find_crt_paths() -> Option<CrtPaths> {
    let search_dirs = [
        Path::new("/usr/lib64"),
        Path::new("/usr/lib/x86_64-linux-gnu"), // Debian/Ubuntu
        Path::new("/usr/lib"),
    ];

    let dynamic_linker_candidates = [
        Path::new("/lib64/ld-linux-x86-64.so.2"),
        Path::new("/usr/lib64/ld-linux-x86-64.so.2"),
        Path::new("/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"), // Debian/Ubuntu
    ];

    let mut crt1 = None;
    let mut crti = None;
    let mut crtn = None;

    for dir in &search_dirs {
        let c1 = dir.join("crt1.o");
        let ci = dir.join("crti.o");
        let cn = dir.join("crtn.o");
        if c1.exists() && ci.exists() && cn.exists() {
            crt1 = Some(c1);
            crti = Some(ci);
            crtn = Some(cn);
            break;
        }
    }

    let dynamic_linker = dynamic_linker_candidates
        .iter()
        .find(|p| p.exists())
        .map(|p| p.to_path_buf());

    match (crt1, crti, crtn, dynamic_linker) {
        (Some(crt1), Some(crti), Some(crtn), Some(dl)) => Some(CrtPaths {
            crt1,
            crti,
            crtn,
            dynamic_linker: dl,
        }),
        _ => None,
    }
}
