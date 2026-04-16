mod addr_analysis;
pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod link;
pub mod parser;

pub use error::TinyErr;

fn default_opts() -> blitz::compile::CompileOptions {
    blitz::compile::CompileOptions::o1()
}

/// Run the tinyc frontend: tokenize, parse, and generate IR.
fn frontend(src: &str) -> Result<codegen::Codegen, TinyErr> {
    let tokens = lexer::tokenize(src)?;
    let program = parser::Parser::parse(tokens)?;
    codegen::Codegen::generate(&program)
}

/// Compile a TinyC source string to object file bytes.
pub fn compile_source(src: &str) -> Result<Vec<u8>, TinyErr> {
    compile_source_with_opts(src, &default_opts())
}

/// Compile a TinyC source string to object file bytes with custom options.
pub fn compile_source_with_opts(
    src: &str,
    opts: &blitz::compile::CompileOptions,
) -> Result<Vec<u8>, TinyErr> {
    let cg = frontend(src)?;
    let obj = blitz::compile::compile_module_with_globals(
        cg.functions,
        opts,
        cg.globals,
        cg.rodata,
        cg.extern_globals,
    )?;
    Ok(obj.finalize())
}

/// Compile a TinyC source string and return the raw ObjectFile (for disassembly).
pub fn compile_to_object(src: &str) -> Result<blitz::emit::object::ObjectFile, TinyErr> {
    compile_to_object_with_opts(src, &default_opts())
}

/// Compile a TinyC source string and return the raw ObjectFile with custom options.
pub fn compile_to_object_with_opts(
    src: &str,
    opts: &blitz::compile::CompileOptions,
) -> Result<blitz::emit::object::ObjectFile, TinyErr> {
    let cg = frontend(src)?;
    let obj = blitz::compile::compile_module_with_globals(
        cg.functions,
        opts,
        cg.globals,
        cg.rodata,
        cg.extern_globals,
    )?;
    Ok(obj)
}

/// Compile a TinyC source string to IR text.
pub fn compile_to_ir(src: &str) -> Result<String, TinyErr> {
    compile_to_ir_with_opts(src, &default_opts())
}

/// Compile a TinyC source string to IR text with custom options.
pub fn compile_to_ir_with_opts(
    src: &str,
    opts: &blitz::compile::CompileOptions,
) -> Result<String, TinyErr> {
    let cg = frontend(src)?;
    let ir = blitz::compile::compile_module_to_ir(cg.functions, opts)?;
    Ok(ir)
}

#[cfg(test)]
mod tests;
