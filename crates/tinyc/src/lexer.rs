use crate::error::TinyErr;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Int,
    If,
    Else,
    While,
    Return,
    IntLit(i64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Bang,
    Assign,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

pub fn tokenize(input: &str) -> Result<Vec<SpannedToken>, TinyErr> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0usize;
    let mut line = 1u32;
    let mut col = 1u32;

    while pos < chars.len() {
        let ch = chars[pos];

        // Skip whitespace
        if ch.is_whitespace() {
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
            pos += 1;
            continue;
        }

        // Skip line comments
        if ch == '/' && pos + 1 < chars.len() && chars[pos + 1] == '/' {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
            continue;
        }

        let span = Span { line, col };

        let tok = match ch {
            '+' => {
                pos += 1;
                col += 1;
                Token::Plus
            }
            '-' => {
                pos += 1;
                col += 1;
                Token::Minus
            }
            '*' => {
                pos += 1;
                col += 1;
                Token::Star
            }
            '/' => {
                pos += 1;
                col += 1;
                Token::Slash
            }
            '%' => {
                pos += 1;
                col += 1;
                Token::Percent
            }
            '(' => {
                pos += 1;
                col += 1;
                Token::LParen
            }
            ')' => {
                pos += 1;
                col += 1;
                Token::RParen
            }
            '{' => {
                pos += 1;
                col += 1;
                Token::LBrace
            }
            '}' => {
                pos += 1;
                col += 1;
                Token::RBrace
            }
            ',' => {
                pos += 1;
                col += 1;
                Token::Comma
            }
            ';' => {
                pos += 1;
                col += 1;
                Token::Semi
            }
            '=' => {
                if pos + 1 < chars.len() && chars[pos + 1] == '=' {
                    pos += 2;
                    col += 2;
                    Token::Eq
                } else {
                    pos += 1;
                    col += 1;
                    Token::Assign
                }
            }
            '!' => {
                if pos + 1 < chars.len() && chars[pos + 1] == '=' {
                    pos += 2;
                    col += 2;
                    Token::Ne
                } else {
                    pos += 1;
                    col += 1;
                    Token::Bang
                }
            }
            '<' => {
                if pos + 1 < chars.len() && chars[pos + 1] == '=' {
                    pos += 2;
                    col += 2;
                    Token::Le
                } else {
                    pos += 1;
                    col += 1;
                    Token::Lt
                }
            }
            '>' => {
                if pos + 1 < chars.len() && chars[pos + 1] == '=' {
                    pos += 2;
                    col += 2;
                    Token::Ge
                } else {
                    pos += 1;
                    col += 1;
                    Token::Gt
                }
            }
            '&' => {
                if pos + 1 < chars.len() && chars[pos + 1] == '&' {
                    pos += 2;
                    col += 2;
                    Token::And
                } else {
                    return Err(TinyErr {
                        line,
                        col,
                        msg: "expected '&&'".to_string(),
                    });
                }
            }
            '|' => {
                if pos + 1 < chars.len() && chars[pos + 1] == '|' {
                    pos += 2;
                    col += 2;
                    Token::Or
                } else {
                    return Err(TinyErr {
                        line,
                        col,
                        msg: "expected '||'".to_string(),
                    });
                }
            }
            c if c.is_ascii_digit() => {
                let start = pos;
                while pos < chars.len() && chars[pos].is_ascii_digit() {
                    pos += 1;
                    col += 1;
                }
                let s: String = chars[start..pos].iter().collect();
                let val: i64 = s.parse().map_err(|_| TinyErr {
                    line,
                    col: span.col,
                    msg: format!("invalid integer literal '{s}'"),
                })?;
                Token::IntLit(val)
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = pos;
                while pos < chars.len() && (chars[pos].is_alphanumeric() || chars[pos] == '_') {
                    pos += 1;
                    col += 1;
                }
                let s: String = chars[start..pos].iter().collect();
                match s.as_str() {
                    "int" => Token::Int,
                    "if" => Token::If,
                    "else" => Token::Else,
                    "while" => Token::While,
                    "return" => Token::Return,
                    _ => Token::Ident(s),
                }
            }
            other => {
                return Err(TinyErr {
                    line,
                    col,
                    msg: format!("unexpected character '{other}'"),
                });
            }
        };

        tokens.push(SpannedToken { token: tok, span });
    }

    tokens.push(SpannedToken {
        token: Token::Eof,
        span: Span { line, col },
    });

    Ok(tokens)
}
