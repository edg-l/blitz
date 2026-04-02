mod addr_analysis;
pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod link;
pub mod parser;

pub use error::TinyErr;

fn default_opts() -> blitz::compile::CompileOptions {
    blitz::compile::CompileOptions {
        enable_inlining: true,
        ..Default::default()
    }
}

/// Compile a TinyC source string to object file bytes.
pub fn compile_source(src: &str) -> Result<Vec<u8>, TinyErr> {
    let tokens = lexer::tokenize(src)?;
    let program = parser::Parser::parse(tokens)?;
    let cg = codegen::Codegen::generate(&program)?;
    let obj = blitz::compile::compile_module(cg.functions, &default_opts())?;
    Ok(obj.finalize())
}

/// Compile a TinyC source string and return the raw ObjectFile (for disassembly).
pub fn compile_to_object(src: &str) -> Result<blitz::emit::object::ObjectFile, TinyErr> {
    let tokens = lexer::tokenize(src)?;
    let program = parser::Parser::parse(tokens)?;
    let cg = codegen::Codegen::generate(&program)?;
    let obj = blitz::compile::compile_module(cg.functions, &default_opts())?;
    Ok(obj)
}

/// Compile a TinyC source string to IR text.
pub fn compile_to_ir(src: &str) -> Result<String, TinyErr> {
    let tokens = lexer::tokenize(src)?;
    let program = parser::Parser::parse(tokens)?;
    let cg = codegen::Codegen::generate(&program)?;
    let ir = blitz::compile::compile_module_to_ir(cg.functions, &default_opts())?;
    Ok(ir)
}

#[cfg(test)]
mod tests;
