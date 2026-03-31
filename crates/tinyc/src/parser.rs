use crate::ast::{BinOp, Expr, FnDef, Program, Stmt, UnaryOp};
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
        while !p.at(Token::Eof) {
            functions.push(p.parse_function()?);
        }
        Ok(Program { functions })
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

    fn parse_function(&mut self) -> Result<FnDef, TinyErr> {
        // int name(int p1, int p2, ...) { body }
        self.expect(Token::Int)?;
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
            self.expect(Token::Int)?;
            let p = self.advance().clone();
            match &p.token {
                Token::Ident(s) => params.push(s.clone()),
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
        Ok(FnDef { name, params, body })
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
        match self.peek().clone() {
            Token::Return => {
                self.advance();
                let e = self.parse_expr()?;
                self.expect(Token::Semi)?;
                Ok(Stmt::Return(e))
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
            Token::Int => {
                // Variable declaration: int name = expr;
                self.advance();
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
                Ok(Stmt::VarDecl { name, init })
            }
            Token::Ident(name) => {
                // Could be assignment or expression statement
                // Look ahead: if next token is Assign, it's an assignment
                if self.pos + 1 < self.tokens.len()
                    && self.tokens[self.pos + 1].token == Token::Assign
                {
                    self.advance(); // consume ident
                    self.advance(); // consume '='
                    let expr = self.parse_expr()?;
                    self.expect(Token::Semi)?;
                    Ok(Stmt::Assign { name, expr })
                } else {
                    let e = self.parse_expr()?;
                    self.expect(Token::Semi)?;
                    Ok(Stmt::ExprStmt(e))
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

    // Pratt parser: precedences
    // || -> 1, && -> 2, == != -> 3, < > <= >= -> 4, + - -> 5, * / % -> 6
    fn infix_bp(tok: &Token) -> Option<(u8, u8)> {
        match tok {
            Token::Or => Some((1, 2)),
            Token::And => Some((3, 4)),
            Token::Eq | Token::Ne => Some((5, 6)),
            Token::Lt | Token::Gt | Token::Le | Token::Ge => Some((7, 8)),
            Token::Plus | Token::Minus => Some((9, 10)),
            Token::Star | Token::Slash | Token::Percent => Some((11, 12)),
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, TinyErr> {
        let mut lhs = self.parse_prefix()?;

        while let Some((l_bp, r_bp)) = Self::infix_bp(self.peek()) {
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
                _ => unreachable!(),
            };
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }

        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr, TinyErr> {
        match self.peek().clone() {
            Token::Minus => {
                self.advance();
                let expr = self.parse_expr_bp(13)?; // higher than any binary op
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            Token::Bang => {
                self.advance();
                let expr = self.parse_expr_bp(13)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
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
                self.advance();
                let e = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(e)
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
