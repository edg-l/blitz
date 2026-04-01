// Usage: tinyc <input.c> [-o <output>] [--emit-ir] [--emit-asm]
// Compiles a .c file to a native executable via Blitz backend + ld/cc linker.

use std::io::Write;
use std::process::exit;

enum Mode {
    Compile,
    EmitIr,
    EmitAsm,
}

fn main() {
    blitz::trace::init_tracing();
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: tinyc <input.c> [-o <output>] [--emit-ir] [--emit-asm]");
        exit(1);
    }

    let mut input_path = None;
    let mut output_path = "a.out".to_string();
    let mut mode = Mode::Compile;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--emit-ir" => {
                mode = Mode::EmitIr;
                i += 1;
            }
            "--emit-asm" => {
                mode = Mode::EmitAsm;
                i += 1;
            }
            "-o" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("tinyc: -o requires an argument");
                    exit(1);
                }
            }
            _ => {
                if input_path.is_none() {
                    input_path = Some(args[i].clone());
                }
                i += 1;
            }
        }
    }

    let input_path = input_path.unwrap_or_else(|| {
        eprintln!("Usage: tinyc <input.c> [-o <output>] [--emit-ir] [--emit-asm]");
        exit(1);
    });

    // Read source file
    let src = std::fs::read_to_string(&input_path).unwrap_or_else(|e| {
        eprintln!("tinyc: cannot read '{}': {}", input_path, e);
        exit(1);
    });

    match mode {
        Mode::EmitIr => {
            let ir = tinyc::compile_to_ir(&src).unwrap_or_else(|e| {
                eprintln!("tinyc: {}", e);
                exit(1);
            });
            print!("{}", ir);
        }
        Mode::EmitAsm => {
            let obj = tinyc::compile_to_object(&src).unwrap_or_else(|e| {
                eprintln!("tinyc: {}", e);
                exit(1);
            });
            for fi in &obj.functions {
                println!("# {}", fi.name);
                let code = &obj.code[fi.offset..fi.offset + fi.size];
                match blitz::test_utils::objdump_disasm(code) {
                    Some(disasm) => print!("{}", disasm),
                    None => {
                        eprintln!("tinyc: objdump not found, cannot disassemble");
                        exit(1);
                    }
                }
            }
        }
        Mode::Compile => {
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

            // Link object file into executable
            let output = std::path::Path::new(&output_path);
            if let Err(e) = tinyc::link::link(&tmp_obj, output) {
                let _ = std::fs::remove_file(&tmp_obj);
                eprintln!("tinyc: {}", e);
                exit(1);
            }

            let _ = std::fs::remove_file(&tmp_obj);
        }
    }
}
