use crate::ast::{BinOp, CType, Expr, ExternDecl, FnDef, Program, Stmt, UnaryOp};
use crate::error::TinyErr;
use crate::lexer::{Span, SpannedToken, Token};

pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    pub fn parse(tokens: Vec<SpannedToken>) -> Result<Program, TinyErr> {
        let mut p = Parser { tokens, pos: 0 };
        let mut functions = Vec::new();
        let mut extern_decls = Vec::new();
        while !p.at(Token::Eof) {
            if p.at(Token::Extern) {
                extern_decls.push(p.parse_extern_decl()?);
            } else {
                functions.push(p.parse_function()?);
            }
        }
        Ok(Program {
            functions,
            extern_decls,
        })
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn span(&self) -> &Span {
        &self.tokens[self.pos].span
    }

    fn at(&self, tok: Token) -> bool {
        self.peek() == &tok
    }

    fn advance(&mut self) -> &SpannedToken {
        let t = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, tok: Token) -> Result<&SpannedToken, TinyErr> {
        if self.peek() == &tok {
            Ok(self.advance())
        } else {
            let span = self.span().clone();
            Err(TinyErr {
                line: span.line,
                col: span.col,
                msg: format!("expected {tok:?}, got {:?}", self.peek()),
            })
        }
    }

    /// Returns true if the current token starts a type specifier.
    fn peek_is_type(&self) -> bool {
        matches!(
            self.peek(),
            Token::Void | Token::Char | Token::Short | Token::Int | Token::Long | Token::Unsigned
        )
    }

    /// Parse a type specifier, advancing past all consumed tokens.
    fn parse_type(&mut self) -> Result<CType, TinyErr> {
        let base = match self.peek().clone() {
            Token::Void => {
                self.advance();
                CType::Void
            }
            Token::Char => {
                self.advance();
                CType::Char
            }
            Token::Short => {
                self.advance();
                CType::Short
            }
            Token::Int => {
                self.advance();
                CType::Int
            }
            Token::Long => {
                self.advance();
                CType::Long
            }
            Token::Unsigned => {
                self.advance();
                match self.peek() {
                    Token::Char => {
                        self.advance();
                        CType::UChar
                    }
                    Token::Short => {
                        self.advance();
                        CType::UShort
                    }
                    Token::Int => {
                        self.advance();
                        CType::UInt
                    }
                    Token::Long => {
                        self.advance();
                        CType::ULong
                    }
                    // Bare `unsigned` == `unsigned int` per C standard.
                    _ => CType::UInt,
                }
            }
            other => {
                let span = self.span().clone();
                return Err(TinyErr {
                    line: span.line,
                    col: span.col,
                    msg: format!("expected type, got {other:?}"),
                });
            }
        };

        // Consume trailing `*` tokens to build pointer types.
        let mut ty = base;
        while self.at(Token::Star) {
            self.advance();
            ty = CType::Ptr(Box::new(ty));
        }
        Ok(ty)
    }

    fn parse_extern_decl(&mut self) -> Result<ExternDecl, TinyErr> {
        self.expect(Token::Extern)?;
        let return_type = self.parse_type()?;
        let name_tok = self.advance().clone();
        let name = match &name_tok.token {
            Token::Ident(s) => s.clone(),
            other => {
                return Err(TinyErr {
                    line: name_tok.span.line,
                    col: name_tok.span.col,
                    msg: format!("expected function name, got {other:?}"),
                });
            }
        };
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        while !self.at(Token::RParen) {
            let param_type = self.parse_type()?;
            // Optional parameter name — discard if present
            if let Token::Ident(_) = self.peek() {
                self.advance();
            }
            params.push(param_type);
            if self.at(Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(Token::RParen)?;
        self.expect(Token::Semi)?;
        Ok(ExternDecl {
            name,
            return_type,
            params,
        })
    }

    fn parse_function(&mut self) -> Result<FnDef, TinyErr> {
        // <type> name(<type> p1, <type> p2, ...) { body }
        let return_type = self.parse_type()?;

        let name_tok = self.advance().clone();
        let name = match &name_tok.token {
            Token::Ident(s) => s.clone(),
            other => {
                return Err(TinyErr {
                    line: name_tok.span.line,
                    col: name_tok.span.col,
                    msg: format!("expected function name, got {other:?}"),
                });
            }
        };

        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        while !self.at(Token::RParen) {
            let param_type = self.parse_type()?;
            let p = self.advance().clone();
            match &p.token {
                Token::Ident(s) => params.push((param_type, s.clone())),
                other => {
                    return Err(TinyErr {
                        line: p.span.line,
                        col: p.span.col,
                        msg: format!("expected parameter name, got {other:?}"),
                    });
                }
            }
            if self.at(Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(Token::RParen)?;

        let body = self.parse_block()?;
        Ok(FnDef {
            name,
            return_type,
            params,
            body,
        })
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, TinyErr> {
        self.expect(Token::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(Token::RBrace) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(Token::RBrace)?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, TinyErr> {
        // Check for type-started variable declaration first.
        if self.peek_is_type() {
            let ty = self.parse_type()?;
            let name_tok = self.advance().clone();
            let name = match &name_tok.token {
                Token::Ident(s) => s.clone(),
                other => {
                    return Err(TinyErr {
                        line: name_tok.span.line,
                        col: name_tok.span.col,
                        msg: format!("expected variable name, got {other:?}"),
                    });
                }
            };
            self.expect(Token::Assign)?;
            let init = self.parse_expr()?;
            self.expect(Token::Semi)?;
            return Ok(Stmt::VarDecl { ty, name, init });
        }

        match self.peek().clone() {
            Token::Return => {
                self.advance();
                if self.at(Token::Semi) {
                    self.advance();
                    Ok(Stmt::Return(None))
                } else {
                    let e = self.parse_expr()?;
                    self.expect(Token::Semi)?;
                    Ok(Stmt::Return(Some(e)))
                }
            }
            Token::If => {
                self.advance();
                self.expect(Token::LParen)?;
                let cond = self.parse_expr()?;
                self.expect(Token::RParen)?;
                let then_body = self.parse_block()?;
                let else_body = if self.at(Token::Else) {
                    self.advance();
                    Some(self.parse_block()?)
                } else {
                    None
                };
                Ok(Stmt::If {
                    cond,
                    then_body,
                    else_body,
                })
            }
            Token::While => {
                self.advance();
                self.expect(Token::LParen)?;
                let cond = self.parse_expr()?;
                self.expect(Token::RParen)?;
                let body = self.parse_block()?;
                Ok(Stmt::While { cond, body })
            }
            Token::Star => {
                // Could be `*expr = value;` (deref assign) or `*expr;` (expr stmt).
                let expr = self.parse_expr()?;
                if self.at(Token::Assign) {
                    self.advance();
                    let value = self.parse_expr()?;
                    self.expect(Token::Semi)?;
                    Ok(Stmt::DerefAssign {
                        addr_expr: expr,
                        value,
                    })
                } else {
                    self.expect(Token::Semi)?;
                    Ok(Stmt::ExprStmt(expr))
                }
            }
            Token::Ident(_) => {
                // Parse LHS expression, then decide: assign, index assign, or expr stmt.
                let lhs = self.parse_expr()?;
                if self.at(Token::Assign) {
                    self.advance();
                    let value = self.parse_expr()?;
                    self.expect(Token::Semi)?;
                    match lhs {
                        Expr::Var(name) => Ok(Stmt::Assign { name, expr: value }),
                        Expr::Index { base, index } => Ok(Stmt::IndexAssign {
                            base: *base,
                            index: *index,
                            value,
                        }),
                        _ => {
                            let span = self.span().clone();
                            Err(TinyErr {
                                line: span.line,
                                col: span.col,
                                msg: "invalid assignment target".to_string(),
                            })
                        }
                    }
                } else {
                    self.expect(Token::Semi)?;
                    Ok(Stmt::ExprStmt(lhs))
                }
            }
            _ => {
                let e = self.parse_expr()?;
                self.expect(Token::Semi)?;
                Ok(Stmt::ExprStmt(e))
            }
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr, TinyErr> {
        self.parse_expr_bp(0)
    }

    // Pratt parser with C-standard precedence levels:
    // ||            -> 1,2
    // &&            -> 3,4
    // | (bitwise)   -> 5,6
    // ^ (bitwise)   -> 7,8
    // & (bitwise)   -> 9,10
    // == !=         -> 11,12
    // < > <= >=     -> 13,14
    // << >>         -> 15,16
    // + -           -> 17,18
    // * / %         -> 19,20
    fn infix_bp(tok: &Token) -> Option<(u8, u8)> {
        match tok {
            Token::Or => Some((1, 2)),
            Token::And => Some((3, 4)),
            Token::Pipe => Some((5, 6)),
            Token::Caret => Some((7, 8)),
            Token::Amp => Some((9, 10)),
            Token::Eq | Token::Ne => Some((11, 12)),
            Token::Lt | Token::Gt | Token::Le | Token::Ge => Some((13, 14)),
            Token::Shl | Token::Shr => Some((15, 16)),
            Token::Plus | Token::Minus => Some((17, 18)),
            Token::Star | Token::Slash | Token::Percent => Some((19, 20)),
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, TinyErr> {
        let mut lhs = self.parse_prefix()?;

        loop {
            // Postfix `[index]` binds at bp 25, tighter than any binary op.
            if self.at(Token::LBracket) {
                if 25 <= min_bp {
                    break;
                }
                self.advance();
                let index = self.parse_expr()?;
                self.expect(Token::RBracket)?;
                lhs = Expr::Index {
                    base: Box::new(lhs),
                    index: Box::new(index),
                };
                continue;
            }

            // Infix binary operators.
            if let Some((l_bp, r_bp)) = Self::infix_bp(self.peek()) {
                if l_bp <= min_bp {
                    break;
                }
                let op_tok = self.advance().token.clone();
                let rhs = self.parse_expr_bp(r_bp)?;
                let op = match op_tok {
                    Token::Plus => BinOp::Add,
                    Token::Minus => BinOp::Sub,
                    Token::Star => BinOp::Mul,
                    Token::Slash => BinOp::Div,
                    Token::Percent => BinOp::Mod,
                    Token::Eq => BinOp::Eq,
                    Token::Ne => BinOp::Ne,
                    Token::Lt => BinOp::Lt,
                    Token::Gt => BinOp::Gt,
                    Token::Le => BinOp::Le,
                    Token::Ge => BinOp::Ge,
                    Token::And => BinOp::And,
                    Token::Or => BinOp::Or,
                    Token::Amp => BinOp::BitAnd,
                    Token::Pipe => BinOp::BitOr,
                    Token::Caret => BinOp::BitXor,
                    Token::Shl => BinOp::Shl,
                    Token::Shr => BinOp::Shr,
                    _ => unreachable!(),
                };
                lhs = Expr::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
                continue;
            }

            break;
        }

        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr, TinyErr> {
        match self.peek().clone() {
            Token::Minus => {
                self.advance();
                let expr = self.parse_expr_bp(21)?; // higher than any binary op
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            Token::Bang => {
                self.advance();
                let expr = self.parse_expr_bp(21)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            Token::Tilde => {
                self.advance();
                let expr = self.parse_expr_bp(21)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::BitNot,
                    expr: Box::new(expr),
                })
            }
            Token::Star => {
                self.advance();
                let expr = self.parse_expr_bp(21)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    expr: Box::new(expr),
                })
            }
            Token::Amp => {
                self.advance();
                let expr = self.parse_expr_bp(21)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(expr),
                })
            }
            Token::Sizeof => {
                self.advance();
                self.expect(Token::LParen)?;
                let ty = self.parse_type()?;
                self.expect(Token::RParen)?;
                Ok(Expr::Sizeof(ty))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, TinyErr> {
        match self.peek().clone() {
            Token::IntLit(v) => {
                self.advance();
                Ok(Expr::IntLit(v))
            }
            Token::StringLit(bytes) => {
                self.advance();
                Ok(Expr::StringLit(bytes))
            }
            Token::Ident(name) => {
                self.advance();
                // Check for function call
                if self.at(Token::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.at(Token::RParen) {
                        args.push(self.parse_expr()?);
                        if self.at(Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Token::LParen => {
                // Disambiguate: `(type)expr` (cast) vs `(expr)` (grouping).
                // Type keywords are distinct tokens so we can check pos+1.
                let next_is_type = self.pos + 1 < self.tokens.len()
                    && matches!(
                        self.tokens[self.pos + 1].token,
                        Token::Void
                            | Token::Char
                            | Token::Short
                            | Token::Int
                            | Token::Long
                            | Token::Unsigned
                    );
                if next_is_type {
                    self.advance(); // consume '('
                    let ty = self.parse_type()?;
                    self.expect(Token::RParen)?;
                    let expr = self.parse_expr_bp(21)?; // unary-level
                    Ok(Expr::Cast {
                        ty,
                        expr: Box::new(expr),
                    })
                } else {
                    self.advance(); // consume '('
                    let e = self.parse_expr()?;
                    self.expect(Token::RParen)?;
                    Ok(e)
                }
            }
            other => {
                let span = self.span().clone();
                Err(TinyErr {
                    line: span.line,
                    col: span.col,
                    msg: format!("unexpected token {other:?}"),
                })
            }
        }
    }
}
