use std::fmt;

#[derive(Debug)]
pub struct TinyErr {
    pub line: u32,
    pub col: u32,
    pub msg: String,
}

impl fmt::Display for TinyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.msg)
    }
}

impl std::error::Error for TinyErr {}

impl From<blitz::compile::CompileError> for TinyErr {
    fn from(e: blitz::compile::CompileError) -> Self {
        TinyErr {
            line: 0,
            col: 0,
            msg: e.to_string(),
        }
    }
}
