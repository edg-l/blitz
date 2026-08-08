use crate::emit::object::{FunctionInfo, ObjectFile};
use crate::ir::function::Function;
use crate::verify;

use super::{CompileError, CompileOptions, compile};

/// Verify every function in the module at a module-level pass boundary.
/// No-op unless `BLITZ_VERIFY` is set.
fn verify_all(stage: &str, functions: &[Function]) {
    if !verify::is_enabled() {
        return;
    }
    for func in functions {
        if let Some(egraph) = func.egraph.as_ref() {
            verify::verify_stage(stage, func, egraph);
        }
    }
}

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
    verify_all("ir-construction", &functions);

    crate::inline::inline_module(&mut functions, opts);
    verify_all("inlining", &functions);

    // DCE1: remove unreachable blocks created by inlining. Unconditional, for
    // the same reason DCE2's CFG half is: an unreachable block is not an
    // optimization opportunity, it is work the rest of the pipeline would do
    // for code that cannot run.
    for func in &mut functions {
        super::dce::run_dce1(func);
    }
    verify_all("dce1", &functions);

    // A call to a pure function whose results nobody reads computes nothing that
    // can be observed. Module-level because purity is: a function is pure when it
    // stores nothing and calls nothing impure, and `printf` is impure because
    // nothing here can see what it does.
    //
    // Gated with the dead-load half of DCE and for the same reason: removing a
    // call takes away something a debugger could have stepped into, which is the
    // line `-O0` holds. The CFG half of DCE runs at every level; this does not.
    if opts.enable_dce {
        let pure = super::dce::pure_functions(&functions);
        let mut removed = 0;
        for func in &mut functions {
            let Some(egraph) = func.egraph.take() else {
                continue;
            };
            removed += super::dce::eliminate_dead_pure_calls(func, &egraph, &pure);
            func.egraph = Some(egraph);
        }
        if removed > 0 && crate::trace::is_enabled("dce") {
            eprintln!("[dce] eliminated {removed} dead call(s) to pure function(s)");
        }
        verify_all("dead-pure-calls", &functions);
    }

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

        // Every function starts on a 16-byte boundary.
        //
        // Not a micro-optimization of the entry itself: it is what makes any
        // statement about where code *inside* a function sits mean anything.
        // Without it a function begins wherever the previous one ended -- measured
        // before this, the four functions of `live/tail_recursion.c` started at
        // offsets 0, 13, 14 and 14 mod 16 -- so a loop's distance from a fetch
        // boundary is decided by the length of everything emitted before it.
        //
        // That is not a theoretical concern here. Deleting 4 dead instructions
        // from `live/matmul_rt.c` cost `+20.7%` cycles, and three instructions of
        // unrelated padding took the same comparison to `-1.0%`; see the loop
        // alignment item in `ROADMAP.md`. Padding *between* functions cannot
        // affect any distance within one, so it carries none of the ordering
        // hazard that aligning a loop header does.
        const FUNC_ALIGN: usize = 16;
        let pad = (FUNC_ALIGN - combined_code.len() % FUNC_ALIGN) % FUNC_ALIGN;
        combined_code.resize(combined_code.len() + pad, 0x90);

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::FunctionBuilder;
    use crate::ir::types::Type;

    fn adder(name: &str, extra: usize) -> Function {
        let mut b = FunctionBuilder::new(name, &[Type::I64], &[Type::I64]);
        let p = b.params().to_vec();
        let mut acc = p[0];
        // `extra` controls the function's length, which is the whole point of the
        // test below: it is what would shift everything after it.
        for k in 0..=extra {
            let c = b.iconst(k as i64 + 1, Type::I64);
            acc = b.add(acc, c);
        }
        b.ret(Some(acc));
        b.finalize().expect("finalize")
    }

    /// Every function starts on a 16-byte boundary, whatever precedes it.
    ///
    /// This is what makes a statement about where code sits *inside* a function
    /// mean anything: with it, a function's offsets modulo 16 do not depend on the
    /// length of any function before it, so a codegen change in one cannot re-time
    /// the loops of another. Measured without it, growing the first of four
    /// functions moved every absolute address in the last one modulo 16.
    #[test]
    fn function_starts_are_16_byte_aligned() {
        for extra in [0usize, 1, 3, 7] {
            let obj = compile_module(
                vec![adder("first", extra), adder("second", 2), adder("third", 5)],
                &CompileOptions::o1(),
            )
            .expect("module compiles");
            assert_eq!(obj.functions.len(), 3);
            for fi in &obj.functions {
                assert_eq!(
                    fi.offset % 16,
                    0,
                    "function '{}' starts at offset {} with extra={extra}, which is not \
                     16-byte aligned",
                    fi.name,
                    fi.offset,
                );
            }
        }
    }
}
