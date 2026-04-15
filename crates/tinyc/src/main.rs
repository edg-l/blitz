// Usage: tinyc <input.c> [input2.c ...] [-o <output>] [-c] [--emit-ir] [--emit-asm]
// Compiles one or more .c files to a native executable via Blitz backend + ld/cc linker.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::exit;

enum Mode {
    Compile,
    CompileOnly,
    EmitIr,
    EmitAsm,
}

fn usage() -> ! {
    eprintln!(
        "Usage: tinyc <input.c> [input2.c ...] [-o <output>] [-c] [--emit-ir] [--emit-asm] [--enable-licm]"
    );
    exit(1);
}

fn main() {
    blitz::trace::init_tracing();
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        usage();
    }

    let mut input_paths: Vec<String> = Vec::new();
    let mut output_path = "a.out".to_string();
    let mut mode = Mode::Compile;
    let mut compile_only = false;
    let mut enable_licm = false;

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
            "--enable-licm" => {
                enable_licm = true;
                i += 1;
            }
            "-c" => {
                compile_only = true;
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
                input_paths.push(args[i].clone());
                i += 1;
            }
        }
    }

    let opts = blitz::compile::CompileOptions {
        enable_inlining: true,
        enable_licm,
        ..Default::default()
    };

    if input_paths.is_empty() {
        usage();
    }

    // Apply compile_only flag to mode
    if compile_only {
        mode = Mode::CompileOnly;
    }

    // Validate flag combinations
    if matches!(mode, Mode::EmitIr | Mode::EmitAsm) && input_paths.len() > 1 {
        eprintln!("tinyc: --emit-ir/--emit-asm only supported with a single input file");
        exit(1);
    }

    if compile_only && output_path != "a.out" && input_paths.len() > 1 {
        eprintln!("tinyc: -o with -c and multiple input files is ambiguous");
        exit(1);
    }

    match mode {
        Mode::EmitIr => {
            let input = &input_paths[0];
            let src = read_source(input);
            let ir = tinyc::compile_to_ir_with_opts(&src, &opts).unwrap_or_else(|e| {
                eprintln!("tinyc: {}: {}", input, e);
                exit(1);
            });
            print!("{}", ir);
        }
        Mode::EmitAsm => {
            let input = &input_paths[0];
            let src = read_source(input);
            let obj = tinyc::compile_to_object_with_opts(&src, &opts).unwrap_or_else(|e| {
                eprintln!("tinyc: {}: {}", input, e);
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
        Mode::CompileOnly => {
            for input in &input_paths {
                let src = read_source(input);
                let obj_bytes =
                    tinyc::compile_source_with_opts(&src, &opts).unwrap_or_else(|e| {
                        eprintln!("tinyc: {}: {}", input, e);
                        exit(1);
                    });
                // Determine output path: use -o if given (single file), else derive from input
                let dest = if output_path != "a.out" {
                    PathBuf::from(&output_path)
                } else {
                    derive_obj_path(input)
                };
                write_file(&dest, &obj_bytes, input);
            }
        }
        Mode::Compile => {
            let pid = std::process::id();
            let tmp_dir = std::env::temp_dir();
            let mut tmp_objs: Vec<PathBuf> = Vec::new();

            // Compile each input to a temp object file
            for (idx, input) in input_paths.iter().enumerate() {
                let src = read_source(input);
                let obj_bytes =
                    tinyc::compile_source_with_opts(&src, &opts).unwrap_or_else(|e| {
                        cleanup(&tmp_objs);
                        eprintln!("tinyc: {}: {}", input, e);
                        exit(1);
                    });
                let tmp_obj = tmp_dir.join(format!("tinyc_{pid}_{idx}.o"));
                write_file(&tmp_obj, &obj_bytes, input);
                tmp_objs.push(tmp_obj);
            }

            // Link all object files into the output executable
            let output = Path::new(&output_path);
            if let Err(e) = tinyc::link::link(&tmp_objs, output) {
                cleanup(&tmp_objs);
                eprintln!("tinyc: {}", e);
                exit(1);
            }

            cleanup(&tmp_objs);
        }
    }
}

fn read_source(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("tinyc: cannot read '{}': {}", path, e);
        exit(1);
    })
}

/// Derive a .o output path from a .c input path: `foo.c` -> `foo.o`.
fn derive_obj_path(input: &str) -> PathBuf {
    let p = Path::new(input);
    let stem = p.file_stem().unwrap_or_else(|| std::ffi::OsStr::new(input));
    let mut out = p.with_file_name(stem);
    out.set_extension("o");
    out
}

fn write_file(dest: &Path, bytes: &[u8], input: &str) {
    let mut f = std::fs::File::create(dest).unwrap_or_else(|e| {
        eprintln!(
            "tinyc: {}: cannot create output file '{}': {}",
            input,
            dest.display(),
            e
        );
        exit(1);
    });
    f.write_all(bytes).unwrap_or_else(|e| {
        eprintln!(
            "tinyc: {}: cannot write output file '{}': {}",
            input,
            dest.display(),
            e
        );
        exit(1);
    });
}

fn cleanup(paths: &[PathBuf]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}
