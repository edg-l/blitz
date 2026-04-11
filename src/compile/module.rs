use crate::emit::object::{FunctionInfo, ObjectFile};
use crate::ir::function::Function;

use super::{CompileError, CompileOptions, compile};

/// Compile multiple functions into a single object file.
///
/// Each `Function` (with its embedded e-graph) is consumed and compiled independently.
pub fn compile_module(
    functions: Vec<Function>,
    opts: &CompileOptions,
) -> Result<ObjectFile, CompileError> {
    compile_module_with_globals(functions, opts, vec![], vec![], vec![])
}

/// Compile multiple functions into a single object file, with global variable definitions.
pub fn compile_module_with_globals(
    mut functions: Vec<Function>,
    opts: &CompileOptions,
    globals: Vec<crate::emit::object::GlobalInfo>,
    rodata: Vec<crate::emit::object::GlobalInfo>,
    extern_globals: Vec<String>,
) -> Result<ObjectFile, CompileError> {
    let has_main = functions.iter().any(|f| f.name == "main");
    crate::inline::inline_module(&mut functions, opts, has_main);

    // Collect global and rodata names so we can filter them from externals.
    let global_names: std::collections::HashSet<String> = globals
        .iter()
        .chain(rodata.iter())
        .map(|g| g.name.clone())
        .collect();

    let mut combined_code: Vec<u8> = Vec::new();
    let mut combined_relocs = Vec::new();
    let mut combined_funcs: Vec<FunctionInfo> = Vec::new();
    let mut combined_externals: Vec<String> = Vec::new();

    for func in functions {
        let obj = compile(func, opts, None)?;

        // Adjust relocation offsets by the current combined code offset.
        let base_offset = combined_code.len();
        for mut reloc in obj.relocations {
            reloc.offset += base_offset;
            combined_relocs.push(reloc);
        }

        // Adjust function offsets.
        for mut fi in obj.functions {
            fi.offset += base_offset;
            combined_funcs.push(fi);
        }

        combined_code.extend_from_slice(&obj.code);

        // Collect unique externals, excluding global variable names.
        for ext in obj.externals {
            if !combined_externals.contains(&ext) && !global_names.contains(&ext) {
                combined_externals.push(ext);
            }
        }
    }

    // Add extern globals as undefined symbols.
    for name in extern_globals {
        if !combined_externals.contains(&name) && !global_names.contains(&name) {
            combined_externals.push(name);
        }
    }

    Ok(ObjectFile {
        code: combined_code,
        relocations: combined_relocs,
        functions: combined_funcs,
        externals: combined_externals,
        globals,
        rodata,
    })
}
