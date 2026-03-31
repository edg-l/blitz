// Usage: tinyc <input.c> [-o <output>]
// Compiles a .c file to a native executable via Blitz backend + cc linker.

use std::io::Write;
use std::process::{Command, exit};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: tinyc <input.c> [-o <output>]");
        exit(1);
    }

    let input_path = &args[1];

    // Parse -o flag
    let output_path = {
        let mut out = None;
        let mut i = 2;
        while i < args.len() {
            if args[i] == "-o" && i + 1 < args.len() {
                out = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        }
        out.unwrap_or_else(|| "a.out".to_string())
    };

    // Read source file
    let src = std::fs::read_to_string(input_path).unwrap_or_else(|e| {
        eprintln!("tinyc: cannot read '{}': {}", input_path, e);
        exit(1);
    });

    // Compile to object bytes
    let obj_bytes = tinyc::compile_source(&src).unwrap_or_else(|e| {
        eprintln!("tinyc: {}", e);
        exit(1);
    });

    let pid = std::process::id();
    let tmp_dir = std::env::temp_dir();
    let tmp_obj = tmp_dir.join(format!("tinyc_{pid}.o"));

    {
        let mut f = std::fs::File::create(&tmp_obj).unwrap_or_else(|e| {
            eprintln!("tinyc: cannot create temp object file: {}", e);
            exit(1);
        });
        f.write_all(&obj_bytes).unwrap_or_else(|e| {
            eprintln!("tinyc: cannot write object file: {}", e);
            exit(1);
        });
    }

    // Link with cc (no runtime helpers needed — division uses native IDIV)
    let status = Command::new("cc")
        .arg(&tmp_obj)
        .arg("-o")
        .arg(&output_path)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("tinyc: cannot run cc: {}", e);
            let _ = std::fs::remove_file(&tmp_obj);
            exit(1);
        });

    let _ = std::fs::remove_file(&tmp_obj);

    if !status.success() {
        eprintln!("tinyc: linker failed");
        exit(1);
    }
}
