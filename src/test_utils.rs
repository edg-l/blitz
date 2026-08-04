#![allow(dead_code)]
/// CI note: `nasm` and `binutils` must be installed for full encoding verification.
/// Tests that use these tools will skip gracefully if the tool is absent.
use std::process::Command;

/// Returns true if the given tool is available on PATH.
pub fn has_tool(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Disassemble bytes using ndisasm in 64-bit mode.
/// Returns None if ndisasm is not available.
pub fn ndisasm(bytes: &[u8]) -> Option<String> {
    if !has_tool("ndisasm") {
        return None;
    }
    use std::io::Write;
    let mut child = Command::new("ndisasm")
        .args(["-b", "64", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take().unwrap().write_all(bytes).ok()?;
    let output = child.wait_with_output().ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        None
    }
}

/// Disassemble bytes using objdump in x86-64 mode.
/// Returns None if objdump is not available.
pub fn objdump_disasm(bytes: &[u8]) -> Option<String> {
    if !has_tool("objdump") {
        return None;
    }
    // A temp file per thread AND per process. The thread id alone is unique only
    // within one process, and every `tinyc` run is single-threaded, so two
    // concurrent processes both picked `ThreadId(1)` and disassembled whichever
    // one wrote the shared file last -- which reads as one compiler emitting
    // another's code.
    let tid = std::thread::current().id();
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("blitz_test_{pid}_{tid:?}.bin"));
    std::fs::write(&tmp, bytes).ok()?;
    let output = Command::new("objdump")
        .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel"])
        .arg(&tmp)
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&tmp);
    if !output.status.success() {
        return None;
    }
    // objdump names its input file in a header line, so keeping it would make the
    // disassembly of one program depend on which process disassembled it. Two
    // runs of the same compiler over the same source must produce the same text:
    // that equality is what says a change to the compiler changed no output.
    let text = String::from_utf8_lossy(&output.stdout);
    Some(
        text.lines()
            .filter(|line| !line.contains("file format binary"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}
