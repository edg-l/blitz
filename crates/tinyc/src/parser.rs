use crate::ast::{BinOp, CType, Expr, ExternDecl, FnDef, GlobalVar, Program, Stmt, UnaryOp};
use crate::error::TinyErr;
use crate::lexer::{Span, SpannedToken, Token};

enum FnOrForward {
    Fn(FnDef),
    Forward(ExternDecl),
}

pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    pub fn parse(tokens: Vec<SpannedToken>) -> Result<Program, TinyErr> {
        let mut p = Parser { tokens, pos: 0 };
        let mut functions = Vec::new();
        let mut extern_decls = Vec::new();
        let mut struct_defs = Vec::new();
        let mut global_vars = Vec::new();
        while !p.at(Token::Eof) {
            if p.at(Token::Extern) {
                extern_decls.push(p.parse_extern_decl()?);
            } else if p.at(Token::Struct)
                && p.pos + 2 < p.tokens.len()
                && p.tokens[p.pos + 2].token == Token::LBrace
            {
                struct_defs.push(p.parse_struct_def()?);
            } else {
                let noinline = p.try_parse_attribute_noinline()?;

                // Save position to disambiguate function vs global variable.
                let save_pos = p.pos;
                let ty = p.parse_type()?;

                let name_tok = p.advance().clone();
                let name = match &name_tok.token {
                    Token::Ident(s) => s.clone(),
                    other => {
                        return Err(TinyErr {
                            line: name_tok.span.line,
                            col: name_tok.span.col,
                            msg: format!("expected identifier, got {other:?}"),
                        });
                    }
                };

                if p.at(Token::LParen) {
                    // Function definition or forward declaration: restore and re-parse.
                    p.pos = save_pos;
                    match p.parse_function_or_forward_decl(noinline)? {
                        FnOrForward::Fn(f) => functions.push(f),
                        FnOrForward::Forward(d) => extern_decls.push(d),
                    }
                } else {
                    // Global variable declaration.
                    if noinline {
                        return Err(TinyErr {
                            line: name_tok.span.line,
                            col: name_tok.span.col,
                            msg: "__attribute__((noinline)) cannot be applied to a global variable"
                                .into(),
                        });
                    }
                    // Parse optional array dimensions.
                    let ty = p.parse_array_dims(ty)?;

                    // Parse optional initializer.
                    let init = if p.at(Token::Assign) {
                        p.advance();
                        let init_tok = p.advance().clone();
                        match &init_tok.token {
                            Token::IntLit(n) => Some(*n),
                            other => {
                                return Err(TinyErr {
                                    line: init_tok.span.line,
                                    col: init_tok.span.col,
                                    msg: format!(
                                        "global initializer must be an integer constant, got {other:?}"
                                    ),
                                });
                            }
                        }
                    } else {
                        None
                    };

                    p.expect(Token::Semi)?;
                    global_vars.push(GlobalVar { name, ty, init });
                }
            }
        }
        let global_vars_opt = if global_vars.is_empty() {
            None
        } else {
            Some(global_vars)
        };
        Ok(Program {
            functions,
            extern_decls,
            struct_defs,
            global_vars: global_vars_opt,
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
            Token::Void
                | Token::Char
                | Token::Short
                | Token::Int
                | Token::Long
                | Token::Unsigned
                | Token::Struct
        )
    }

    /// Parse zero or more `[N]` array dimension suffixes, wrapping `ty` in CType::Array.
    fn parse_array_dims(&mut self, mut ty: CType) -> Result<CType, TinyErr> {
        while self.at(Token::LBracket) {
            self.advance();
            let dim_tok = self.advance().clone();
            let dim = match &dim_tok.token {
                Token::IntLit(n) if *n > 0 => *n as usize,
                Token::IntLit(n) => {
                    return Err(TinyErr {
                        line: dim_tok.span.line,
                        col: dim_tok.span.col,
                        msg: format!("array size must be positive, got {n}"),
                    });
                }
                other => {
                    return Err(TinyErr {
                        line: dim_tok.span.line,
                        col: dim_tok.span.col,
                        msg: format!("expected array size, got {other:?}"),
                    });
                }
            };
            self.expect(Token::RBracket)?;
            ty = CType::Array(Box::new(ty), dim);
        }
        Ok(ty)
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
            Token::Struct => {
                self.advance();
                let name_tok = self.advance().clone();
                match &name_tok.token {
                    Token::Ident(s) => CType::Struct(s.clone()),
                    other => {
                        return Err(TinyErr {
                            line: name_tok.span.line,
                            col: name_tok.span.col,
                            msg: format!("expected struct name, got {other:?}"),
                        });
                    }
                }
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
        // Parse array dimensions in type context (e.g. sizeof(int[10]))
        let ty = self.parse_array_dims(ty)?;
        Ok(ty)
    }

    fn parse_struct_def(&mut self) -> Result<(String, Vec<(String, CType)>), TinyErr> {
        self.expect(Token::Struct)?;
        let name_tok = self.advance().clone();
        let name = match &name_tok.token {
            Token::Ident(s) => s.clone(),
            other => {
                return Err(TinyErr {
                    line: name_tok.span.line,
                    col: name_tok.span.col,
                    msg: format!("expected struct name, got {other:?}"),
                });
            }
        };
        self.expect(Token::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(Token::RBrace) {
            let field_ty = self.parse_type()?;
            let field_tok = self.advance().clone();
            let field_name = match &field_tok.token {
                Token::Ident(s) => s.clone(),
                other => {
                    return Err(TinyErr {
                        line: field_tok.span.line,
                        col: field_tok.span.col,
                        msg: format!("expected field name, got {other:?}"),
                    });
                }
            };
            // Parse array dimensions on struct fields
            let field_ty = self.parse_array_dims(field_ty)?;
            self.expect(Token::Semi)?;
            fields.push((field_name, field_ty));
        }
        self.expect(Token::RBrace)?;
        self.expect(Token::Semi)?;
        Ok((name, fields))
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
            let mut param_type = self.parse_type()?;
            // Optional parameter name -- discard if present
            if let Token::Ident(_) = self.peek() {
                self.advance();
                // Parse array dimensions and decay to pointer
                param_type = self.parse_array_dims(param_type)?.decay();
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

    /// Try to parse `__attribute__((noinline))`. Returns true if found.
    fn try_parse_attribute_noinline(&mut self) -> Result<bool, TinyErr> {
        if !matches!(self.peek(), Token::Ident(s) if s == "__attribute__") {
            return Ok(false);
        }
        self.advance(); // __attribute__
        self.expect(Token::LParen)?;
        self.expect(Token::LParen)?;
        let attr_tok = self.advance().clone();
        match &attr_tok.token {
            Token::Ident(s) if s == "noinline" => {}
            other => {
                return Err(TinyErr {
                    line: attr_tok.span.line,
                    col: attr_tok.span.col,
                    msg: format!("unsupported attribute: {other:?}"),
                });
            }
        }
        self.expect(Token::RParen)?;
        self.expect(Token::RParen)?;
        Ok(true)
    }

    fn parse_function_or_forward_decl(&mut self, noinline: bool) -> Result<FnOrForward, TinyErr> {
        // <type> name(<params>) { body }   -- function definition
        // <type> name(<params>) ;          -- forward declaration
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
            let mut param_type = self.parse_type()?;
            let p = self.advance().clone();
            match &p.token {
                Token::Ident(s) => {
                    // Parse array dimensions and decay to pointer
                    param_type = self.parse_array_dims(param_type)?.decay();
                    params.push((param_type, s.clone()));
                }
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

        // Forward declaration: semicolon instead of body
        if self.at(Token::Semi) {
            self.advance();
            let param_types = params.into_iter().map(|(ty, _)| ty).collect();
            return Ok(FnOrForward::Forward(ExternDecl {
                name,
                return_type,
                params: param_types,
            }));
        }

        let body = self.parse_block()?;
        Ok(FnOrForward::Fn(FnDef {
            name,
            return_type,
            params,
            body,
            noinline,
        }))
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, TinyErr> {
        self.expect(Token::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(Token::RBrace) {
            if self.at(Token::For) {
                let for_stmts = self.parse_for()?;
                stmts.extend(for_stmts);
            } else {
                stmts.push(self.parse_stmt()?);
            }
        }
        self.expect(Token::RBrace)?;
        Ok(stmts)
    }

    /// Parse `for(init; cond; update) { body }`.
    fn parse_for(&mut self) -> Result<Vec<Stmt>, TinyErr> {
        self.advance(); // consume `for`
        self.expect(Token::LParen)?;

        // init: variable declaration, expression statement, or empty
        let init = if self.at(Token::Semi) {
            self.advance();
            None
        } else if self.peek_is_type() {
            let stmt = self.parse_stmt()?; // parses VarDecl including trailing `;`
            Some(stmt)
        } else {
            let stmt = self.parse_expr_or_assign()?;
            self.expect(Token::Semi)?;
            Some(stmt)
        };

        // cond: expression or empty (infinite loop)
        let cond = if self.at(Token::Semi) {
            self.advance();
            Expr::IntLit(1) // always true
        } else {
            let c = self.parse_expr()?;
            self.expect(Token::Semi)?;
            c
        };

        // update: assignment or expression or empty (no trailing semicolon)
        let update = if self.at(Token::RParen) {
            None
        } else {
            Some(self.parse_expr_or_assign()?)
        };
        self.expect(Token::RParen)?;

        let body = self.parse_block()?;

        // The init is a separate statement before the for loop, so we return
        // a Vec that may contain [init, for] or just [for].
        let mut result = Vec::new();
        if let Some(init_stmt) = init {
            result.push(init_stmt);
        }
        result.push(Stmt::For {
            init: None, // init already extracted above
            cond,
            update: update.map(Box::new),
            body,
        });
        Ok(result)
    }

    /// Parse an expression, then optionally an `= value` turning it into an assignment statement.
    fn parse_expr_or_assign(&mut self) -> Result<Stmt, TinyErr> {
        let expr = self.parse_expr()?;
        if self.at(Token::Assign) {
            self.advance();
            let value = self.parse_expr()?;
            match expr {
                Expr::Var(name) => Ok(Stmt::Assign { name, expr: value }),
                Expr::Index { base, index } => Ok(Stmt::IndexAssign {
                    base: *base,
                    index: *index,
                    value,
                }),
                Expr::FieldAccess { expr: e, field } => Ok(Stmt::FieldAssign {
                    expr: *e,
                    field,
                    value,
                }),
                Expr::UnaryOp {
                    op: UnaryOp::Deref, ..
                } => Ok(Stmt::DerefAssign {
                    addr_expr: expr,
                    value,
                }),
                _ => {
                    let span = self.span().clone();
                    Err(TinyErr {
                        line: span.line,
                        col: span.col,
                        msg: "invalid assignment target".into(),
                    })
                }
            }
        } else {
            Ok(Stmt::ExprStmt(expr))
        }
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
            // Parse array dimensions: e.g. `int arr[3][4]`
            let ty = self.parse_array_dims(ty)?;
            if ty.is_array() && self.at(Token::Assign) {
                let span = self.span().clone();
                return Err(TinyErr {
                    line: span.line,
                    col: span.col,
                    msg: "array initializers are not supported".into(),
                });
            }
            if self.at(Token::Semi) {
                self.advance();
                return Ok(Stmt::VarDecl {
                    ty,
                    name,
                    init: None,
                });
            }
            self.expect(Token::Assign)?;
            let init = self.parse_expr()?;
            self.expect(Token::Semi)?;
            return Ok(Stmt::VarDecl {
                ty,
                name,
                init: Some(init),
            });
        }

        match self.peek().clone() {
            Token::Break => {
                self.advance();
                self.expect(Token::Semi)?;
                Ok(Stmt::Break)
            }
            Token::Continue => {
                self.advance();
                self.expect(Token::Semi)?;
                Ok(Stmt::Continue)
            }
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
                // Parse LHS expression, then decide: assign, index assign, field assign, or expr stmt.
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
                        Expr::FieldAccess { expr, field } => Ok(Stmt::FieldAssign {
                            expr: *expr,
                            field,
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
            // Postfix operators bind at bp 25, tighter than any binary op.
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

            // Postfix `.field`
            if self.at(Token::Dot) {
                if 25 <= min_bp {
                    break;
                }
                self.advance();
                let field_tok = self.advance().clone();
                let field = match &field_tok.token {
                    Token::Ident(s) => s.clone(),
                    other => {
                        return Err(TinyErr {
                            line: field_tok.span.line,
                            col: field_tok.span.col,
                            msg: format!("expected field name after '.', got {other:?}"),
                        });
                    }
                };
                lhs = Expr::FieldAccess {
                    expr: Box::new(lhs),
                    field,
                };
                continue;
            }

            // Postfix `->field` desugars to `(*lhs).field`
            if self.at(Token::Arrow) {
                if 25 <= min_bp {
                    break;
                }
                self.advance();
                let field_tok = self.advance().clone();
                let field = match &field_tok.token {
                    Token::Ident(s) => s.clone(),
                    other => {
                        return Err(TinyErr {
                            line: field_tok.span.line,
                            col: field_tok.span.col,
                            msg: format!("expected field name after '->', got {other:?}"),
                        });
                    }
                };
                lhs = Expr::FieldAccess {
                    expr: Box::new(Expr::UnaryOp {
                        op: UnaryOp::Deref,
                        expr: Box::new(lhs),
                    }),
                    field,
                };
                continue;
            }

            // Ternary operator: cond ? then : else (right-associative, below ||)
            if self.at(Token::Question) {
                if 1 <= min_bp {
                    break;
                }
                self.advance();
                let then_expr = self.parse_expr()?;
                self.expect(Token::Colon)?;
                let else_expr = self.parse_expr_bp(0)?;
                lhs = Expr::Ternary {
                    cond: Box::new(lhs),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
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
                            | Token::Struct
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
