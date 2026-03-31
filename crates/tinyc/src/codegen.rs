use std::collections::HashMap;

use blitz::ir::builder::{FunctionBuilder, Value, Variable};
use blitz::ir::condcode::CondCode;
use blitz::ir::function::Function;
use blitz::ir::types::Type;

use crate::ast::{BinOp, Expr, FnDef, Program, Stmt, UnaryOp};
use crate::error::TinyErr;

pub struct Codegen {
    pub functions: Vec<Function>,
}

impl Codegen {
    pub fn generate(program: &Program) -> Result<Codegen, TinyErr> {
        let mut functions = Vec::new();
        for func in &program.functions {
            functions.push(compile_fn(func)?);
        }
        Ok(Codegen { functions })
    }
}

struct FnCtx<'b> {
    builder: &'b mut FunctionBuilder,
    locals: HashMap<String, Variable>,
}

impl<'b> FnCtx<'b> {
    fn new(builder: &'b mut FunctionBuilder) -> Self {
        FnCtx {
            builder,
            locals: HashMap::new(),
        }
    }

    fn is_terminated(&self) -> bool {
        self.builder.is_current_block_terminated()
    }

    /// Convert an i64 condition value to Flags for use in branch.
    fn val_to_flags(&mut self, val: Value) -> Value {
        let zero = self.builder.const_i64(0);
        self.builder.icmp(CondCode::Ne, val, zero)
    }

    /// Compile an expression as a branch condition, returning Flags directly.
    ///
    /// For comparison expressions (BinOp with a relational op), this returns the
    /// icmp flags directly to avoid a select->icmp roundtrip that would cause flag
    /// clobbering between the comparison and the branch.
    ///
    /// For other expressions, falls back to val_to_flags(compile_expr(...)).
    fn compile_cond(&mut self, expr: &Expr) -> Result<Value, TinyErr> {
        match expr {
            Expr::BinOp { op, lhs, rhs } => {
                let cc = match op {
                    BinOp::Eq => Some(CondCode::Eq),
                    BinOp::Ne => Some(CondCode::Ne),
                    BinOp::Lt => Some(CondCode::Slt),
                    BinOp::Gt => Some(CondCode::Sgt),
                    BinOp::Le => Some(CondCode::Sle),
                    BinOp::Ge => Some(CondCode::Sge),
                    _ => None,
                };
                if let Some(cc) = cc {
                    let l = self.compile_expr(lhs)?;
                    let r = self.compile_expr(rhs)?;
                    Ok(self.builder.icmp(cc, l, r))
                } else {
                    let val = self.compile_expr(expr)?;
                    Ok(self.val_to_flags(val))
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::Not,
                expr: inner,
            } => {
                // !x branches when x == 0, so use Eq instead of Ne
                let val = self.compile_expr(inner)?;
                let zero = self.builder.const_i64(0);
                Ok(self.builder.icmp(CondCode::Eq, val, zero))
            }
            _ => {
                let val = self.compile_expr(expr)?;
                Ok(self.val_to_flags(val))
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<Value, TinyErr> {
        match expr {
            Expr::IntLit(v) => Ok(self.builder.const_i64(*v)),
            Expr::Var(name) => {
                let var = self.locals.get(name).copied().ok_or_else(|| TinyErr {
                    line: 0,
                    col: 0,
                    msg: format!("undefined variable '{name}'"),
                })?;
                Ok(self.builder.use_var(var))
            }
            Expr::UnaryOp { op, expr } => {
                let val = self.compile_expr(expr)?;
                match op {
                    UnaryOp::Neg => Ok(self.builder.neg(val)),
                    UnaryOp::Not => {
                        // !x == (x == 0)
                        let zero = self.builder.const_i64(0);
                        Ok(self.builder.icmp_val(CondCode::Eq, val, zero))
                    }
                }
            }
            Expr::BinOp { op, lhs, rhs } => {
                match op {
                    // Short-circuit logical operators
                    BinOp::And => {
                        // a && b: (l != 0) & (r != 0) as integer
                        let l = self.compile_expr(lhs)?;
                        let r = self.compile_expr(rhs)?;
                        let zero = self.builder.const_i64(0);
                        let lf = self.builder.icmp_val(CondCode::Ne, l, zero);
                        let zero2 = self.builder.const_i64(0);
                        let rf = self.builder.icmp_val(CondCode::Ne, r, zero2);
                        Ok(self.builder.and(lf, rf))
                    }
                    BinOp::Or => {
                        let l = self.compile_expr(lhs)?;
                        let r = self.compile_expr(rhs)?;
                        let zero = self.builder.const_i64(0);
                        let lf = self.builder.icmp_val(CondCode::Ne, l, zero);
                        let zero2 = self.builder.const_i64(0);
                        let rf = self.builder.icmp_val(CondCode::Ne, r, zero2);
                        Ok(self.builder.or(lf, rf))
                    }
                    _ => {
                        let l = self.compile_expr(lhs)?;
                        let r = self.compile_expr(rhs)?;
                        match op {
                            BinOp::Add => Ok(self.builder.add(l, r)),
                            BinOp::Sub => Ok(self.builder.sub(l, r)),
                            BinOp::Mul => Ok(self.builder.mul(l, r)),
                            BinOp::Div => Ok(self.builder.sdiv(l, r)),
                            BinOp::Mod => Ok(self.builder.srem(l, r)),
                            BinOp::Eq => Ok(self.builder.icmp_val(CondCode::Eq, l, r)),
                            BinOp::Ne => Ok(self.builder.icmp_val(CondCode::Ne, l, r)),
                            BinOp::Lt => Ok(self.builder.icmp_val(CondCode::Slt, l, r)),
                            BinOp::Gt => Ok(self.builder.icmp_val(CondCode::Sgt, l, r)),
                            BinOp::Le => Ok(self.builder.icmp_val(CondCode::Sle, l, r)),
                            BinOp::Ge => Ok(self.builder.icmp_val(CondCode::Sge, l, r)),
                            BinOp::And | BinOp::Or => unreachable!(),
                        }
                    }
                }
            }
            Expr::Call { name, args } => {
                let mut arg_vals = Vec::new();
                for a in args {
                    arg_vals.push(self.compile_expr(a)?);
                }
                let results = self.builder.call(name, &arg_vals, &[Type::I64]);
                Ok(results[0])
            }
        }
    }
}

fn compile_fn(fn_def: &FnDef) -> Result<Function, TinyErr> {
    let param_types: Vec<Type> = fn_def.params.iter().map(|_| Type::I64).collect();
    let mut builder = FunctionBuilder::new(&fn_def.name, &param_types, &[Type::I64]);

    // Bind parameter names as variables
    let param_vals = builder.params().to_vec();
    let mut ctx = FnCtx::new(&mut builder);
    for (name, val) in fn_def.params.iter().zip(param_vals.iter()) {
        let var = ctx.builder.declare_var(Type::I64);
        ctx.builder.def_var(var, *val);
        ctx.locals.insert(name.clone(), var);
    }

    compile_stmts(&mut ctx, &fn_def.body)?;

    // Implicit return 0 if not terminated
    if !ctx.is_terminated() {
        let zero = ctx.builder.const_i64(0);
        ctx.builder.ret(Some(zero));
    }

    builder.finalize().map_err(|e| TinyErr {
        line: 0,
        col: 0,
        msg: e.to_string(),
    })
}

fn compile_stmts(ctx: &mut FnCtx, stmts: &[Stmt]) -> Result<(), TinyErr> {
    for stmt in stmts {
        if ctx.is_terminated() {
            break;
        }
        compile_stmt(ctx, stmt)?;
    }
    Ok(())
}

fn compile_stmt(ctx: &mut FnCtx, stmt: &Stmt) -> Result<(), TinyErr> {
    match stmt {
        Stmt::Return(expr) => {
            let val = ctx.compile_expr(expr)?;
            ctx.builder.ret(Some(val));
        }
        Stmt::ExprStmt(expr) => {
            ctx.compile_expr(expr)?;
        }
        Stmt::VarDecl { name, init } => {
            let val = ctx.compile_expr(init)?;
            let var = ctx.builder.declare_var(Type::I64);
            ctx.builder.def_var(var, val);
            ctx.locals.insert(name.clone(), var);
        }
        Stmt::Assign { name, expr } => {
            let val = ctx.compile_expr(expr)?;
            let var = *ctx.locals.get(name).expect("undefined variable in assign");
            ctx.builder.def_var(var, val);
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            compile_if(ctx, cond, then_body, else_body.as_deref())?;
        }
        Stmt::While { cond, body } => {
            compile_while(ctx, cond, body)?;
        }
    }
    Ok(())
}

fn compile_if(
    ctx: &mut FnCtx,
    cond: &Expr,
    then_body: &[Stmt],
    else_body: Option<&[Stmt]>,
) -> Result<(), TinyErr> {
    let flags = ctx.compile_cond(cond)?;

    let then_block = ctx.builder.create_block();
    let else_block = ctx.builder.create_block();

    ctx.builder.branch(flags, then_block, else_block, &[], &[]);

    // Then
    ctx.builder.set_block(then_block);
    ctx.builder.seal_block(then_block);
    compile_stmts(ctx, then_body)?;
    let then_terminated = ctx.is_terminated();
    let then_exit = ctx.builder.current_block();

    // Else
    ctx.builder.set_block(else_block);
    ctx.builder.seal_block(else_block);
    if let Some(else_stmts) = else_body {
        compile_stmts(ctx, else_stmts)?;
    }
    let else_terminated = ctx.is_terminated();
    let else_exit = ctx.builder.current_block();

    if then_terminated && else_terminated {
        return Ok(());
    }

    // At least one branch falls through -- create merge block.
    let merge_block = ctx.builder.create_block();

    if !then_terminated {
        ctx.builder.set_block(then_exit.unwrap());
        ctx.builder.jump(merge_block, &[]);
    }
    if !else_terminated {
        ctx.builder.set_block(else_exit.unwrap());
        ctx.builder.jump(merge_block, &[]);
    }

    ctx.builder.seal_block(merge_block);
    ctx.builder.set_block(merge_block);

    Ok(())
}

fn compile_while(ctx: &mut FnCtx, cond: &Expr, body: &[Stmt]) -> Result<(), TinyErr> {
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    // Jump to header from current block.
    ctx.builder.jump(header_block, &[]);

    // Header: do NOT seal yet (back edge from body not yet known).
    ctx.builder.set_block(header_block);
    let flags = ctx.compile_cond(cond)?;
    ctx.builder.branch(flags, body_block, exit_block, &[], &[]);

    // Body
    ctx.builder.set_block(body_block);
    ctx.builder.seal_block(body_block);
    compile_stmts(ctx, body)?;
    if !ctx.is_terminated() {
        ctx.builder.jump(header_block, &[]);
    }

    // Now all predecessors of header are known.
    ctx.builder.seal_block(header_block);

    // Exit
    ctx.builder.seal_block(exit_block);
    ctx.builder.set_block(exit_block);

    Ok(())
}
