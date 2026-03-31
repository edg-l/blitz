use std::collections::HashMap;

use blitz::ir::builder::{FunctionBuilder, Value, Variable};
use blitz::ir::condcode::CondCode;
use blitz::ir::function::Function;
use blitz::ir::types::Type;

use crate::ast::{BinOp, CType, Expr, FnDef, Program, Stmt, UnaryOp};
use crate::error::TinyErr;

pub struct Codegen {
    pub functions: Vec<Function>,
}

impl Codegen {
    pub fn generate(program: &Program) -> Result<Codegen, TinyErr> {
        // Pre-scan all function signatures before codegen.
        let mut fn_signatures: HashMap<String, (CType, Vec<CType>)> = HashMap::new();
        for func in &program.functions {
            let param_types: Vec<CType> = func.params.iter().map(|(ty, _)| *ty).collect();
            fn_signatures.insert(func.name.clone(), (func.return_type, param_types));
        }

        let mut functions = Vec::new();
        for func in &program.functions {
            functions.push(compile_fn(func, &fn_signatures)?);
        }
        Ok(Codegen { functions })
    }
}

struct FnCtx<'b> {
    builder: &'b mut FunctionBuilder,
    locals: HashMap<String, Variable>,
    local_types: HashMap<String, CType>,
    fn_return_type: CType,
    fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
}

impl<'b> FnCtx<'b> {
    fn new(
        builder: &'b mut FunctionBuilder,
        fn_return_type: CType,
        fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
    ) -> Self {
        FnCtx {
            builder,
            locals: HashMap::new(),
            local_types: HashMap::new(),
            fn_return_type,
            fn_signatures,
        }
    }

    fn is_terminated(&self) -> bool {
        self.builder.is_current_block_terminated()
    }

    /// Sign-extend, zero-extend, or truncate `val` from `from` to `to`.
    fn emit_convert(&mut self, val: Value, from: CType, to: CType) -> Value {
        if from == to {
            return val;
        }
        let from_w = from.bit_width();
        let to_w = to.bit_width();
        let target = to.to_ir_type().unwrap();
        if to_w > from_w {
            if from.is_signed() {
                self.builder.sext(val, target)
            } else {
                self.builder.zext(val, target)
            }
        } else if to_w < from_w {
            self.builder.trunc(val, target)
        } else {
            // Same width, different signedness: reinterpret, no IR op.
            val
        }
    }

    /// Apply integer promotion (Char/Short/UChar/UShort -> Int).
    fn emit_promote(&mut self, val: Value, ty: CType) -> (Value, CType) {
        let promoted = ty.promoted();
        if promoted != ty {
            let val = self.emit_convert(val, ty, promoted);
            (val, promoted)
        } else {
            (val, ty)
        }
    }

    /// Promote both operands then convert to their usual arithmetic common type.
    fn emit_usual_conversion(
        &mut self,
        lv: Value,
        lt: CType,
        rv: Value,
        rt: CType,
    ) -> (Value, Value, CType) {
        let (lv, lt) = self.emit_promote(lv, lt);
        let (rv, rt) = self.emit_promote(rv, rt);
        let common = CType::usual_arithmetic_conversion(lt, rt);
        let lv = self.emit_convert(lv, lt, common);
        let rv = self.emit_convert(rv, rt, common);
        (lv, rv, common)
    }

    /// Emit icmp+select yielding an I32 0/1 value (C standard: comparison yields int).
    fn emit_icmp_val(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let flags = self.builder.icmp(cc, a, b);
        let one = self.builder.iconst(1, Type::I32);
        let zero = self.builder.iconst(0, Type::I32);
        self.builder.select(flags, one, zero)
    }

    /// Convert a value to Flags using a typed zero constant.
    fn val_to_flags(&mut self, val: Value, ty: CType) -> Value {
        let ir_ty = ty.to_ir_type().unwrap();
        let zero = self.builder.iconst(0, ir_ty);
        self.builder.icmp(CondCode::Ne, val, zero)
    }

    /// Compile an expression as a branch condition, returning Flags directly.
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
                if cc.is_some() {
                    let (l, lt) = self.compile_expr(lhs)?;
                    let (r, rt) = self.compile_expr(rhs)?;
                    let (l, r, common) = self.emit_usual_conversion(l, lt, r, rt);
                    // Pick signed/unsigned condition code.
                    let cc = match op {
                        BinOp::Eq => CondCode::Eq,
                        BinOp::Ne => CondCode::Ne,
                        BinOp::Lt => {
                            if common.is_unsigned() {
                                CondCode::Ult
                            } else {
                                CondCode::Slt
                            }
                        }
                        BinOp::Gt => {
                            if common.is_unsigned() {
                                CondCode::Ugt
                            } else {
                                CondCode::Sgt
                            }
                        }
                        BinOp::Le => {
                            if common.is_unsigned() {
                                CondCode::Ule
                            } else {
                                CondCode::Sle
                            }
                        }
                        BinOp::Ge => {
                            if common.is_unsigned() {
                                CondCode::Uge
                            } else {
                                CondCode::Sge
                            }
                        }
                        _ => unreachable!(),
                    };
                    Ok(self.builder.icmp(cc, l, r))
                } else {
                    let (val, ty) = self.compile_expr(expr)?;
                    Ok(self.val_to_flags(val, ty))
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::Not,
                expr: inner,
            } => {
                let (val, ty) = self.compile_expr(inner)?;
                let (val, ty) = self.emit_promote(val, ty);
                let zero = self.builder.iconst(0, ty.to_ir_type().unwrap());
                Ok(self.builder.icmp(CondCode::Eq, val, zero))
            }
            _ => {
                let (val, ty) = self.compile_expr(expr)?;
                Ok(self.val_to_flags(val, ty))
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(Value, CType), TinyErr> {
        match expr {
            Expr::IntLit(v) => {
                // Unsuffixed decimal literal: i32 if fits, otherwise i64.
                if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                    let val = self.builder.iconst(*v, Type::I32);
                    Ok((val, CType::Int))
                } else {
                    let val = self.builder.iconst(*v, Type::I64);
                    Ok((val, CType::Long))
                }
            }
            Expr::Var(name) => {
                let var = self.locals.get(name).copied().ok_or_else(|| TinyErr {
                    line: 0,
                    col: 0,
                    msg: format!("undefined variable '{name}'"),
                })?;
                let ty = self.local_types[name];
                let val = self.builder.use_var(var);
                Ok((val, ty))
            }
            Expr::UnaryOp { op, expr } => {
                let (val, ty) = self.compile_expr(expr)?;
                match op {
                    UnaryOp::Neg => {
                        let (val, ty) = self.emit_promote(val, ty);
                        Ok((self.builder.neg(val), ty))
                    }
                    UnaryOp::Not => {
                        // !x == (x == 0) -> I32 result
                        let (val, ty) = self.emit_promote(val, ty);
                        let zero = self.builder.iconst(0, ty.to_ir_type().unwrap());
                        Ok((self.emit_icmp_val(CondCode::Eq, val, zero), CType::Int))
                    }
                    UnaryOp::BitNot => {
                        // ~x == x ^ -1
                        let (val, ty) = self.emit_promote(val, ty);
                        let all_ones = self.builder.iconst(-1, ty.to_ir_type().unwrap());
                        Ok((self.builder.xor(val, all_ones), ty))
                    }
                }
            }
            Expr::BinOp { op, lhs, rhs } => {
                match op {
                    BinOp::And => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, lt);
                        let lzero = self.builder.iconst(0, lt.to_ir_type().unwrap());
                        let lbool = self.emit_icmp_val(CondCode::Ne, l, lzero);

                        let (r, rt) = self.compile_expr(rhs)?;
                        let (r, rt) = self.emit_promote(r, rt);
                        let rzero = self.builder.iconst(0, rt.to_ir_type().unwrap());
                        let rbool = self.emit_icmp_val(CondCode::Ne, r, rzero);

                        Ok((self.builder.and(lbool, rbool), CType::Int))
                    }
                    BinOp::Or => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, lt);
                        let lzero = self.builder.iconst(0, lt.to_ir_type().unwrap());
                        let lbool = self.emit_icmp_val(CondCode::Ne, l, lzero);

                        let (r, rt) = self.compile_expr(rhs)?;
                        let (r, rt) = self.emit_promote(r, rt);
                        let rzero = self.builder.iconst(0, rt.to_ir_type().unwrap());
                        let rbool = self.emit_icmp_val(CondCode::Ne, r, rzero);

                        Ok((self.builder.or(lbool, rbool), CType::Int))
                    }
                    // Shift operators: promote independently, result type is promoted left type.
                    BinOp::Shl => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, lt);
                        let (r, _rt) = self.compile_expr(rhs)?;
                        let (r, _rt) = self.emit_promote(r, _rt);
                        // Shift amount must match the left operand's type for the IR.
                        let r = self.emit_convert(r, _rt, lt);
                        Ok((self.builder.shl(l, r), lt))
                    }
                    BinOp::Shr => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, lt);
                        let (r, _rt) = self.compile_expr(rhs)?;
                        let (r, _rt) = self.emit_promote(r, _rt);
                        // Shift amount must match the left operand's type for the IR.
                        let r = self.emit_convert(r, _rt, lt);
                        // sar for signed, shr for unsigned.
                        if lt.is_signed() {
                            Ok((self.builder.sar(l, r), lt))
                        } else {
                            Ok((self.builder.shr(l, r), lt))
                        }
                    }
                    _ => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (r, rt) = self.compile_expr(rhs)?;
                        let (l, r, common) = self.emit_usual_conversion(l, lt, r, rt);
                        match op {
                            BinOp::Add => Ok((self.builder.add(l, r), common)),
                            BinOp::Sub => Ok((self.builder.sub(l, r), common)),
                            BinOp::Mul => Ok((self.builder.mul(l, r), common)),
                            BinOp::Div => {
                                if common.is_unsigned() {
                                    Ok((self.builder.udiv(l, r), common))
                                } else {
                                    Ok((self.builder.sdiv(l, r), common))
                                }
                            }
                            BinOp::Mod => {
                                if common.is_unsigned() {
                                    Ok((self.builder.urem(l, r), common))
                                } else {
                                    Ok((self.builder.srem(l, r), common))
                                }
                            }
                            BinOp::Eq => Ok((self.emit_icmp_val(CondCode::Eq, l, r), CType::Int)),
                            BinOp::Ne => Ok((self.emit_icmp_val(CondCode::Ne, l, r), CType::Int)),
                            BinOp::Lt => {
                                let cc = if common.is_unsigned() {
                                    CondCode::Ult
                                } else {
                                    CondCode::Slt
                                };
                                Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                            }
                            BinOp::Gt => {
                                let cc = if common.is_unsigned() {
                                    CondCode::Ugt
                                } else {
                                    CondCode::Sgt
                                };
                                Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                            }
                            BinOp::Le => {
                                let cc = if common.is_unsigned() {
                                    CondCode::Ule
                                } else {
                                    CondCode::Sle
                                };
                                Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                            }
                            BinOp::Ge => {
                                let cc = if common.is_unsigned() {
                                    CondCode::Uge
                                } else {
                                    CondCode::Sge
                                };
                                Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                            }
                            BinOp::BitAnd => Ok((self.builder.and(l, r), common)),
                            BinOp::BitOr => Ok((self.builder.or(l, r), common)),
                            BinOp::BitXor => Ok((self.builder.xor(l, r), common)),
                            BinOp::And | BinOp::Or | BinOp::Shl | BinOp::Shr => unreachable!(),
                        }
                    }
                }
            }
            Expr::Call { name, args } => {
                let (ret_type, param_types) = self
                    .fn_signatures
                    .get(name.as_str())
                    .map(|(r, p)| (*r, p.clone()))
                    .ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("undefined function '{name}'"),
                    })?;

                // Compile and convert each argument to the expected parameter type.
                let mut arg_vals: Vec<Value> = Vec::new();
                for (i, arg_expr) in args.iter().enumerate() {
                    let (arg_val, arg_ty) = self.compile_expr(arg_expr)?;
                    let param_ty = param_types[i];
                    let converted = self.emit_convert(arg_val, arg_ty, param_ty);
                    arg_vals.push(converted);
                }

                let ret_ir_tys: Vec<Type> =
                    ret_type.to_ir_type().map(|t| vec![t]).unwrap_or_default();
                let results = self.builder.call(name, &arg_vals, &ret_ir_tys);

                if ret_type == CType::Void {
                    // Return a dummy value; void calls shouldn't appear in expression context.
                    let dummy = self.builder.iconst(0, Type::I32);
                    Ok((dummy, CType::Int))
                } else {
                    Ok((results[0], ret_type))
                }
            }
            Expr::Cast { ty, expr } => {
                let (val, from_ty) = self.compile_expr(expr)?;
                let converted = self.emit_convert(val, from_ty, *ty);
                Ok((converted, *ty))
            }
            Expr::Sizeof(ty) => {
                let size = ty.bit_width() as i64 / 8;
                let val = self.builder.iconst(size, Type::I64);
                Ok((val, CType::ULong))
            }
        }
    }
}

fn compile_fn(
    fn_def: &FnDef,
    fn_sigs: &HashMap<String, (CType, Vec<CType>)>,
) -> Result<Function, TinyErr> {
    let param_ir_types: Vec<Type> = fn_def
        .params
        .iter()
        .map(|(ty, _)| ty.to_ir_type().unwrap())
        .collect();
    let ret_ir_types: Vec<Type> = fn_def
        .return_type
        .to_ir_type()
        .map(|t| vec![t])
        .unwrap_or_default();

    let mut builder = FunctionBuilder::new(&fn_def.name, &param_ir_types, &ret_ir_types);

    let param_vals = builder.params().to_vec();
    let mut ctx = FnCtx::new(&mut builder, fn_def.return_type, fn_sigs);

    for ((ty, name), val) in fn_def.params.iter().zip(param_vals.iter()) {
        let var = ctx.builder.declare_var(ty.to_ir_type().unwrap());
        ctx.builder.def_var(var, *val);
        ctx.locals.insert(name.clone(), var);
        ctx.local_types.insert(name.clone(), *ty);
    }

    compile_stmts(&mut ctx, &fn_def.body)?;

    // Implicit return if not terminated.
    if !ctx.is_terminated() {
        if fn_def.return_type == CType::Void {
            ctx.builder.ret(None);
        } else {
            let ir_ty = fn_def.return_type.to_ir_type().unwrap();
            let zero = ctx.builder.iconst(0, ir_ty);
            ctx.builder.ret(Some(zero));
        }
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
        Stmt::Return(opt_expr) => match opt_expr {
            Some(expr) => {
                let (val, ty) = ctx.compile_expr(expr)?;
                let converted = ctx.emit_convert(val, ty, ctx.fn_return_type);
                ctx.builder.ret(Some(converted));
            }
            None => {
                ctx.builder.ret(None);
            }
        },
        Stmt::ExprStmt(expr) => {
            ctx.compile_expr(expr)?;
        }
        Stmt::VarDecl { ty, name, init } => {
            let (val, init_ty) = ctx.compile_expr(init)?;
            let converted = ctx.emit_convert(val, init_ty, *ty);
            let ir_ty = ty.to_ir_type().unwrap();
            let var = ctx.builder.declare_var(ir_ty);
            ctx.builder.def_var(var, converted);
            ctx.locals.insert(name.clone(), var);
            ctx.local_types.insert(name.clone(), *ty);
        }
        Stmt::Assign { name, expr } => {
            let local_ty = *ctx
                .local_types
                .get(name)
                .expect("undefined variable in assign");
            let (val, expr_ty) = ctx.compile_expr(expr)?;
            let converted = ctx.emit_convert(val, expr_ty, local_ty);
            let var = *ctx.locals.get(name).expect("undefined variable in assign");
            ctx.builder.def_var(var, converted);
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
