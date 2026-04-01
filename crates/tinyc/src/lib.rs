mod addr_analysis;
pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod parser;

pub use error::TinyErr;

/// Compile a TinyC source string to object file bytes.
pub fn compile_source(src: &str) -> Result<Vec<u8>, TinyErr> {
    let tokens = lexer::tokenize(src)?;
    let program = parser::Parser::parse(tokens)?;
    let cg = codegen::Codegen::generate(&program)?;
    let opts = blitz::compile::CompileOptions::default();
    let obj = blitz::compile::compile_module(cg.functions, &opts)?;
    Ok(obj.finalize())
}

/// Compile a TinyC source string and return the raw ObjectFile (for disassembly).
pub fn compile_to_object(src: &str) -> Result<blitz::emit::object::ObjectFile, TinyErr> {
    let tokens = lexer::tokenize(src)?;
    let program = parser::Parser::parse(tokens)?;
    let cg = codegen::Codegen::generate(&program)?;
    let opts = blitz::compile::CompileOptions::default();
    let obj = blitz::compile::compile_module(cg.functions, &opts)?;
    Ok(obj)
}

#[cfg(test)]
mod tests;
