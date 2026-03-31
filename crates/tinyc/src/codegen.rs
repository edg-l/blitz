use std::collections::HashMap;

use blitz::ir::builder::{FunctionBuilder, Value};
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
    locals: HashMap<String, Value>,
}

impl<'b> FnCtx<'b> {
    fn new(builder: &'b mut FunctionBuilder) -> Self {
        FnCtx {
            builder,
            locals: HashMap::new(),
        }
    }

    fn set_block(&mut self, block: blitz::ir::effectful::BlockId) {
        self.builder.set_block(block);
    }

    fn is_terminated(&self) -> bool {
        self.builder.is_current_block_terminated()
    }

    fn iconst(&mut self, v: i64) -> Value {
        self.builder.const_i64(v)
    }

    /// Convert an i64 condition value to Flags for use in branch.
    fn val_to_flags(&mut self, val: Value) -> Value {
        let zero = self.iconst(0);
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
                let zero = self.iconst(0);
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
            Expr::IntLit(v) => Ok(self.iconst(*v)),
            Expr::Var(name) => self.locals.get(name).copied().ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: format!("undefined variable '{name}'"),
            }),
            Expr::UnaryOp { op, expr } => {
                let val = self.compile_expr(expr)?;
                match op {
                    UnaryOp::Neg => Ok(self.builder.neg(val)),
                    UnaryOp::Not => {
                        // !x == (x == 0)
                        let zero = self.iconst(0);
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
                        let zero = self.iconst(0);
                        let lf = self.builder.icmp_val(CondCode::Ne, l, zero);
                        let zero2 = self.iconst(0);
                        let rf = self.builder.icmp_val(CondCode::Ne, r, zero2);
                        Ok(self.builder.and(lf, rf))
                    }
                    BinOp::Or => {
                        let l = self.compile_expr(lhs)?;
                        let r = self.compile_expr(rhs)?;
                        let zero = self.iconst(0);
                        let lf = self.builder.icmp_val(CondCode::Ne, l, zero);
                        let zero2 = self.iconst(0);
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

    // Bind parameter names
    let param_vals = builder.params().to_vec();
    let mut ctx = FnCtx::new(&mut builder);
    for (name, val) in fn_def.params.iter().zip(param_vals.iter()) {
        ctx.locals.insert(name.clone(), *val);
    }

    compile_stmts(&mut ctx, &fn_def.body)?;

    // Implicit return 0 if not terminated
    if !ctx.is_terminated() {
        let zero = ctx.iconst(0);
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
            ctx.locals.insert(name.clone(), val);
        }
        Stmt::Assign { name, expr } => {
            let val = ctx.compile_expr(expr)?;
            ctx.locals.insert(name.clone(), val);
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
    // Snapshot locals before branch
    let pre_locals: HashMap<String, Value> = ctx.locals.clone();
    // Collect all live variable names (sorted for deterministic block param order)
    let mut live_vars: Vec<String> = pre_locals.keys().cloned().collect();
    live_vars.sort();

    // Evaluate condition directly as Flags for the branch.
    let flags = ctx.compile_cond(cond)?;

    // Create then/else blocks. Merge block is created later, only if needed.
    let then_block = ctx.builder.create_block();
    let else_block = ctx.builder.create_block();

    // Branch to then/else (merge_block args will be wired after we know its id)
    // We branch without args to then/else; they'll jump to merge with args.
    ctx.builder.branch(flags, then_block, else_block, &[], &[]);

    // Codegen then block
    ctx.set_block(then_block);
    ctx.locals = pre_locals.clone();
    compile_stmts(ctx, then_body)?;
    let then_terminated = ctx.is_terminated();
    let then_locals = ctx.locals.clone();

    // Codegen else block
    ctx.set_block(else_block);
    ctx.locals = pre_locals.clone();
    if let Some(else_stmts) = else_body {
        compile_stmts(ctx, else_stmts)?;
    }
    let else_terminated = ctx.is_terminated();
    let else_locals = ctx.locals.clone();

    if then_terminated && else_terminated {
        // Both branches terminate: no merge needed, no reachable continuation.
        // Mark as terminated; the current block (else_block) is already terminated.

        return Ok(());
    }

    // At least one branch falls through: create merge block with params for live vars.
    let merge_param_types: Vec<Type> = live_vars.iter().map(|_| Type::I64).collect();
    let (merge_block, merge_params) = ctx.builder.create_block_with_params(&merge_param_types);

    // Wire jump from then to merge (if not terminated)
    if !then_terminated {
        ctx.set_block(then_block);
        let then_args: Vec<Value> = live_vars
            .iter()
            .map(|v| then_locals.get(v).copied().unwrap_or_else(|| pre_locals[v]))
            .collect();
        ctx.builder.jump(merge_block, &then_args);
    }

    // Wire jump from else to merge (if not terminated)
    if !else_terminated {
        ctx.set_block(else_block);
        let else_args: Vec<Value> = live_vars
            .iter()
            .map(|v| else_locals.get(v).copied().unwrap_or_else(|| pre_locals[v]))
            .collect();
        ctx.builder.jump(merge_block, &else_args);
    }

    // Continue in merge block with updated locals
    ctx.set_block(merge_block);
    ctx.locals = pre_locals;
    for (var, param_val) in live_vars.iter().zip(merge_params.iter()) {
        ctx.locals.insert(var.clone(), *param_val);
    }

    Ok(())
}

fn compile_while(ctx: &mut FnCtx, cond: &Expr, body: &[Stmt]) -> Result<(), TinyErr> {
    // Snapshot live vars before the loop
    let pre_locals: HashMap<String, Value> = ctx.locals.clone();
    let mut live_vars: Vec<String> = pre_locals.keys().cloned().collect();
    live_vars.sort();

    // Create header block with params for all live vars.
    // Header params serve as the SSA "phi" values for loop variables.
    let header_param_types: Vec<Type> = live_vars.iter().map(|_| Type::I64).collect();
    let (header_block, header_params) = ctx.builder.create_block_with_params(&header_param_types);

    // Create a plain exit block (no params).
    // The loop exit uses header_params directly since they dominate the exit block.
    let exit_block = ctx.builder.create_block();

    // Create body block (no params)
    let body_block = ctx.builder.create_block();

    // Jump from current block to header with initial values
    let init_args: Vec<Value> = live_vars.iter().map(|v| pre_locals[v]).collect();
    ctx.builder.jump(header_block, &init_args);

    // Header block: update locals to header params, evaluate cond, branch
    ctx.set_block(header_block);
    ctx.locals = pre_locals.clone();
    for (var, param_val) in live_vars.iter().zip(header_params.iter()) {
        ctx.locals.insert(var.clone(), *param_val);
    }

    // Evaluate condition directly as Flags for the branch.
    let flags = ctx.compile_cond(cond)?;

    // Branch: body_block if true, exit_block if false (no args to either).
    ctx.builder.branch(flags, body_block, exit_block, &[], &[]);

    // Body block: codegen body, then jump back to header with updated values.
    ctx.set_block(body_block);
    // locals remain as set from header block above (header_params)
    compile_stmts(ctx, body)?;
    let body_terminated = ctx.is_terminated();
    let body_locals = ctx.locals.clone();

    if !body_terminated {
        let back_args: Vec<Value> = live_vars
            .iter()
            .map(|v| body_locals.get(v).copied().unwrap_or_else(|| pre_locals[v]))
            .collect();
        ctx.builder.jump(header_block, &back_args);
    }

    // Continue after the loop in the exit block.
    // Use header_params as the exit values (they hold the values when condition failed).
    ctx.set_block(exit_block);
    // The exit block needs a terminator; emit a dummy ret for now.
    // Actually: the caller will emit code in exit_block. We just need to set up locals.
    ctx.locals = pre_locals;
    for (var, param_val) in live_vars.iter().zip(header_params.iter()) {
        ctx.locals.insert(var.clone(), *param_val);
    }

    Ok(())
}
