use std::collections::{HashMap, HashSet};

use blitz::ir::builder::{FunctionBuilder, Value, Variable};
use blitz::ir::condcode::CondCode;
use blitz::ir::function::{Function, StackSlot};
use blitz::ir::types::Type;

use crate::ast::{BinOp, CType, Expr, FnDef, Program, Stmt, UnaryOp};
use crate::error::TinyErr;

/// Recursively walk all expressions and statements to find variables whose
/// address is taken via `&var_name`.
fn find_addressed_vars(stmts: &[Stmt]) -> HashSet<String> {
    let mut set = HashSet::new();
    for stmt in stmts {
        walk_stmt(stmt, &mut set);
    }
    set
}

fn walk_expr(expr: &Expr, set: &mut HashSet<String>) {
    match expr {
        Expr::UnaryOp {
            op: UnaryOp::AddrOf,
            expr: inner,
        } => {
            if let Expr::Var(name) = inner.as_ref() {
                set.insert(name.clone());
            }
            walk_expr(inner, set);
        }
        Expr::IntLit(_) => {}
        Expr::Var(_) => {}
        Expr::BinOp { lhs, rhs, .. } => {
            walk_expr(lhs, set);
            walk_expr(rhs, set);
        }
        Expr::UnaryOp { expr: inner, .. } => {
            walk_expr(inner, set);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                walk_expr(arg, set);
            }
        }
        Expr::Cast { expr: inner, .. } => {
            walk_expr(inner, set);
        }
        Expr::Sizeof(_) => {}
        Expr::Index { base, index } => {
            walk_expr(base, set);
            walk_expr(index, set);
        }
    }
}

fn walk_stmt(stmt: &Stmt, set: &mut HashSet<String>) {
    match stmt {
        Stmt::Return(Some(expr)) => walk_expr(expr, set),
        Stmt::Return(None) => {}
        Stmt::ExprStmt(expr) => walk_expr(expr, set),
        Stmt::VarDecl { init, .. } => walk_expr(init, set),
        Stmt::Assign { expr, .. } => walk_expr(expr, set),
        Stmt::DerefAssign { addr_expr, value } => {
            walk_expr(addr_expr, set);
            walk_expr(value, set);
        }
        Stmt::IndexAssign { base, index, value } => {
            walk_expr(base, set);
            walk_expr(index, set);
            walk_expr(value, set);
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            walk_expr(cond, set);
            for s in then_body {
                walk_stmt(s, set);
            }
            if let Some(else_stmts) = else_body {
                for s in else_stmts {
                    walk_stmt(s, set);
                }
            }
        }
        Stmt::While { cond, body } => {
            walk_expr(cond, set);
            for s in body {
                walk_stmt(s, set);
            }
        }
    }
}

pub struct Codegen {
    pub functions: Vec<Function>,
}

impl Codegen {
    pub fn generate(program: &Program) -> Result<Codegen, TinyErr> {
        // Pre-scan all function signatures before codegen.
        let mut fn_signatures: HashMap<String, (CType, Vec<CType>)> = HashMap::new();
        for func in &program.functions {
            let param_types: Vec<CType> = func.params.iter().map(|(ty, _)| ty.clone()).collect();
            fn_signatures.insert(func.name.clone(), (func.return_type.clone(), param_types));
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
    stack_slots: HashMap<String, (StackSlot, CType)>,
    addressed_vars: HashSet<String>,
    fn_return_type: CType,
    fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
}

impl<'b> FnCtx<'b> {
    fn new(
        builder: &'b mut FunctionBuilder,
        fn_return_type: CType,
        fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
        addressed_vars: HashSet<String>,
    ) -> Self {
        FnCtx {
            builder,
            locals: HashMap::new(),
            local_types: HashMap::new(),
            stack_slots: HashMap::new(),
            addressed_vars,
            fn_return_type,
            fn_signatures,
        }
    }

    /// Look up the C type for a local variable, checking both stack-allocated
    /// and SSA variables.
    fn local_type(&self, name: &str) -> CType {
        if let Some((_, ty)) = self.stack_slots.get(name) {
            ty.clone()
        } else {
            self.local_types[name].clone()
        }
    }

    fn is_terminated(&self) -> bool {
        self.builder.is_current_block_terminated()
    }

    /// Sign-extend, zero-extend, or truncate `val` from `from` to `to`.
    fn emit_convert(&mut self, val: Value, from: &CType, to: &CType) -> Value {
        if from == to {
            return val;
        }
        // Pointer-to-pointer: identity (both I64).
        if from.is_pointer() && to.is_pointer() {
            return val;
        }
        // Integer-to-pointer: zero-extend to I64 (e.g. NULL assignment).
        if to.is_pointer() && from.is_integer() {
            let from_w = from.bit_width();
            if from_w < 64 {
                return self.builder.zext(val, Type::I64);
            }
            return val;
        }
        // Pointer-to-integer: truncate or reinterpret as needed.
        if from.is_pointer() && to.is_integer() {
            let to_w = to.bit_width();
            if to_w < 64 {
                return self.builder.trunc(val, to.to_ir_type().unwrap());
            }
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
    fn emit_promote(&mut self, val: Value, ty: &CType) -> (Value, CType) {
        let promoted = ty.promoted();
        if promoted != *ty {
            let val = self.emit_convert(val, ty, &promoted);
            (val, promoted)
        } else {
            (val, promoted)
        }
    }

    /// Promote both operands then convert to their usual arithmetic common type.
    fn emit_usual_conversion(
        &mut self,
        lv: Value,
        lt: &CType,
        rv: Value,
        rt: &CType,
    ) -> (Value, Value, CType) {
        let (lv, lt) = self.emit_promote(lv, lt);
        let (rv, rt) = self.emit_promote(rv, rt);
        let common = CType::usual_arithmetic_conversion(&lt, &rt);
        let lv = self.emit_convert(lv, &lt, &common);
        let rv = self.emit_convert(rv, &rt, &common);
        (lv, rv, common)
    }

    /// Emit icmp+select yielding an I32 0/1 value (C standard: comparison yields int).
    fn emit_icmp_val(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let flags = self.builder.icmp(cc, a, b);
        let one = self.builder.iconst(1, Type::I32);
        let zero = self.builder.iconst(0, Type::I32);
        self.builder.select(flags, one, zero)
    }

    /// Emit pointer + integer arithmetic: scale the integer by pointee size, add.
    fn emit_ptr_add(
        &mut self,
        ptr: Value,
        ptr_ty: &CType,
        idx: Value,
        idx_ty: &CType,
    ) -> Result<(Value, CType), TinyErr> {
        if *ptr_ty.pointee() == CType::Void {
            return Err(TinyErr {
                line: 0,
                col: 0,
                msg: "pointer arithmetic on void* is not allowed".into(),
            });
        }
        let elem_size = ptr_ty.pointee_size() as i64;
        let idx = self.emit_convert(idx, idx_ty, &CType::Long);
        let scale = self.builder.iconst(elem_size, Type::I64);
        let offset = self.builder.mul(idx, scale);
        let result = self.builder.add(ptr, offset);
        Ok((result, ptr_ty.clone()))
    }

    /// Emit pointer - integer arithmetic: scale the integer by pointee size, subtract.
    fn emit_ptr_sub(
        &mut self,
        ptr: Value,
        ptr_ty: &CType,
        idx: Value,
        idx_ty: &CType,
    ) -> Result<(Value, CType), TinyErr> {
        if *ptr_ty.pointee() == CType::Void {
            return Err(TinyErr {
                line: 0,
                col: 0,
                msg: "pointer arithmetic on void* is not allowed".into(),
            });
        }
        let elem_size = ptr_ty.pointee_size() as i64;
        let idx = self.emit_convert(idx, idx_ty, &CType::Long);
        let scale = self.builder.iconst(elem_size, Type::I64);
        let offset = self.builder.mul(idx, scale);
        let result = self.builder.sub(ptr, offset);
        Ok((result, ptr_ty.clone()))
    }

    /// Convert a value to Flags using a typed zero constant.
    fn val_to_flags(&mut self, val: Value, ty: &CType) -> Value {
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
                    let (l, r, common) = self.emit_usual_conversion(l, &lt, r, &rt);
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
                    Ok(self.val_to_flags(val, &ty))
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::Not,
                expr: inner,
            } => {
                let (val, ty) = self.compile_expr(inner)?;
                let (val, ty) = self.emit_promote(val, &ty);
                let zero = self.builder.iconst(0, ty.to_ir_type().unwrap());
                Ok(self.builder.icmp(CondCode::Eq, val, zero))
            }
            _ => {
                let (val, ty) = self.compile_expr(expr)?;
                Ok(self.val_to_flags(val, &ty))
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
                if let Some((slot, ty)) = self.stack_slots.get(name) {
                    let slot = *slot;
                    let ty = ty.clone();
                    let ir_ty = ty.to_ir_type().unwrap();
                    let addr = self.builder.stack_addr(slot);
                    let val = self.builder.load(addr, ir_ty);
                    Ok((val, ty))
                } else {
                    let var = self.locals.get(name).copied().ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("undefined variable '{name}'"),
                    })?;
                    let ty = self.local_types[name].clone();
                    let val = self.builder.use_var(var);
                    Ok((val, ty))
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => {
                // &var_name: look up the stack slot for the variable
                if let Expr::Var(name) = inner.as_ref() {
                    let (slot, var_ty) = self.stack_slots.get(name).ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("address-of variable '{name}' not in stack slots"),
                    })?;
                    let slot = *slot;
                    let var_ty = var_ty.clone();
                    let addr = self.builder.stack_addr(slot);
                    Ok((addr, CType::Ptr(Box::new(var_ty))))
                } else {
                    Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: "address-of applied to non-variable expression".into(),
                    })
                }
            }
            Expr::UnaryOp { op, expr } => {
                let (val, ty) = self.compile_expr(expr)?;
                match op {
                    UnaryOp::Neg => {
                        let (val, ty) = self.emit_promote(val, &ty);
                        Ok((self.builder.neg(val), ty))
                    }
                    UnaryOp::Not => {
                        // !x == (x == 0) -> I32 result
                        let (val, ty) = self.emit_promote(val, &ty);
                        let zero = self.builder.iconst(0, ty.to_ir_type().unwrap());
                        Ok((self.emit_icmp_val(CondCode::Eq, val, zero), CType::Int))
                    }
                    UnaryOp::BitNot => {
                        // ~x == x ^ -1
                        let (val, ty) = self.emit_promote(val, &ty);
                        let all_ones = self.builder.iconst(-1, ty.to_ir_type().unwrap());
                        Ok((self.builder.xor(val, all_ones), ty))
                    }
                    UnaryOp::Deref => {
                        // *expr: inner must be a pointer type
                        let pointee = ty.pointee().clone();
                        let ir_ty = pointee.to_ir_type().unwrap();
                        let loaded = self.builder.load(val, ir_ty);
                        Ok((loaded, pointee))
                    }
                    UnaryOp::AddrOf => {
                        unreachable!("AddrOf handled by earlier match arm")
                    }
                }
            }
            Expr::BinOp { op, lhs, rhs } => {
                match op {
                    BinOp::And => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, &lt);
                        let lzero = self.builder.iconst(0, lt.to_ir_type().unwrap());
                        let lbool = self.emit_icmp_val(CondCode::Ne, l, lzero);

                        let (r, rt) = self.compile_expr(rhs)?;
                        let (r, rt) = self.emit_promote(r, &rt);
                        let rzero = self.builder.iconst(0, rt.to_ir_type().unwrap());
                        let rbool = self.emit_icmp_val(CondCode::Ne, r, rzero);

                        Ok((self.builder.and(lbool, rbool), CType::Int))
                    }
                    BinOp::Or => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, &lt);
                        let lzero = self.builder.iconst(0, lt.to_ir_type().unwrap());
                        let lbool = self.emit_icmp_val(CondCode::Ne, l, lzero);

                        let (r, rt) = self.compile_expr(rhs)?;
                        let (r, rt) = self.emit_promote(r, &rt);
                        let rzero = self.builder.iconst(0, rt.to_ir_type().unwrap());
                        let rbool = self.emit_icmp_val(CondCode::Ne, r, rzero);

                        Ok((self.builder.or(lbool, rbool), CType::Int))
                    }
                    // Shift operators: promote independently, result type is promoted left type.
                    BinOp::Shl => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, &lt);
                        let (r, _rt) = self.compile_expr(rhs)?;
                        let (r, _rt) = self.emit_promote(r, &_rt);
                        // Shift amount must match the left operand's type for the IR.
                        let r = self.emit_convert(r, &_rt, &lt);
                        Ok((self.builder.shl(l, r), lt))
                    }
                    BinOp::Shr => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, &lt);
                        let (r, _rt) = self.compile_expr(rhs)?;
                        let (r, _rt) = self.emit_promote(r, &_rt);
                        // Shift amount must match the left operand's type for the IR.
                        let r = self.emit_convert(r, &_rt, &lt);
                        // sar for signed, shr for unsigned.
                        if lt.is_signed() {
                            Ok((self.builder.sar(l, r), lt))
                        } else {
                            Ok((self.builder.shr(l, r), lt))
                        }
                    }
                    // Pointer + integer or integer + pointer arithmetic.
                    BinOp::Add => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (r, rt) = self.compile_expr(rhs)?;
                        if lt.is_pointer() && rt.is_integer() {
                            self.emit_ptr_add(l, &lt, r, &rt)
                        } else if lt.is_integer() && rt.is_pointer() {
                            // Commutative: int + ptr => ptr + int.
                            self.emit_ptr_add(r, &rt, l, &lt)
                        } else {
                            let (l, r, common) = self.emit_usual_conversion(l, &lt, r, &rt);
                            Ok((self.builder.add(l, r), common))
                        }
                    }
                    // Pointer - integer arithmetic.
                    BinOp::Sub => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (r, rt) = self.compile_expr(rhs)?;
                        if lt.is_pointer() && rt.is_integer() {
                            self.emit_ptr_sub(l, &lt, r, &rt)
                        } else {
                            let (l, r, common) = self.emit_usual_conversion(l, &lt, r, &rt);
                            Ok((self.builder.sub(l, r), common))
                        }
                    }
                    _ => {
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (r, rt) = self.compile_expr(rhs)?;
                        let (l, r, common) = self.emit_usual_conversion(l, &lt, r, &rt);
                        match op {
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
                            BinOp::Add
                            | BinOp::Sub
                            | BinOp::And
                            | BinOp::Or
                            | BinOp::Shl
                            | BinOp::Shr => unreachable!(),
                        }
                    }
                }
            }
            Expr::Call { name, args } => {
                let (ret_type, param_types) = self
                    .fn_signatures
                    .get(name.as_str())
                    .map(|(r, p)| (r.clone(), p.clone()))
                    .ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("undefined function '{name}'"),
                    })?;

                // Compile and convert each argument to the expected parameter type.
                let mut arg_vals: Vec<Value> = Vec::new();
                for (i, arg_expr) in args.iter().enumerate() {
                    let (arg_val, arg_ty) = self.compile_expr(arg_expr)?;
                    let param_ty = &param_types[i];
                    let converted = self.emit_convert(arg_val, &arg_ty, param_ty);
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
                let converted = self.emit_convert(val, &from_ty, ty);
                Ok((converted, ty.clone()))
            }
            Expr::Sizeof(ty) => {
                let size = ty.bit_width() as i64 / 8;
                let val = self.builder.iconst(size, Type::I64);
                Ok((val, CType::ULong))
            }
            Expr::Index { base, index } => {
                let (base_val, base_ty) = self.compile_expr(base)?;
                let pointee = base_ty.pointee().clone();
                if pointee == CType::Void {
                    return Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: "pointer arithmetic on void* is not allowed".into(),
                    });
                }
                let elem_size = base_ty.pointee_size() as i64;
                let (idx_val, idx_ty) = self.compile_expr(index)?;
                // Widen index to I64 for address arithmetic
                let idx_val = self.emit_convert(idx_val, &idx_ty, &CType::Long);
                let scale = self.builder.iconst(elem_size, Type::I64);
                let offset = self.builder.mul(idx_val, scale);
                let addr = self.builder.add(base_val, offset);
                let ir_ty = pointee.to_ir_type().unwrap();
                let loaded = self.builder.load(addr, ir_ty);
                Ok((loaded, pointee))
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

    let addressed_vars = find_addressed_vars(&fn_def.body);
    let param_vals = builder.params().to_vec();
    let mut ctx = FnCtx::new(
        &mut builder,
        fn_def.return_type.clone(),
        fn_sigs,
        addressed_vars,
    );

    for ((ty, name), val) in fn_def.params.iter().zip(param_vals.iter()) {
        if ctx.addressed_vars.contains(name) {
            let size = ty.bit_width() / 8;
            let slot = ctx.builder.create_stack_slot(size, 8);
            let addr = ctx.builder.stack_addr(slot);
            ctx.builder.store(addr, *val);
            ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
        } else {
            let var = ctx.builder.declare_var(ty.to_ir_type().unwrap());
            ctx.builder.def_var(var, *val);
            ctx.locals.insert(name.clone(), var);
            ctx.local_types.insert(name.clone(), ty.clone());
        }
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
                let ret_ty = ctx.fn_return_type.clone();
                let converted = ctx.emit_convert(val, &ty, &ret_ty);
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
            let converted = ctx.emit_convert(val, &init_ty, ty);
            if ctx.addressed_vars.contains(name) {
                let size = ty.bit_width() / 8;
                let slot = ctx.builder.create_stack_slot(size, 8);
                let addr = ctx.builder.stack_addr(slot);
                ctx.builder.store(addr, converted);
                ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
            } else {
                let ir_ty = ty.to_ir_type().unwrap();
                let var = ctx.builder.declare_var(ir_ty);
                ctx.builder.def_var(var, converted);
                ctx.locals.insert(name.clone(), var);
                ctx.local_types.insert(name.clone(), ty.clone());
            }
        }
        Stmt::Assign { name, expr } => {
            let local_ty = ctx.local_type(name);
            let (val, expr_ty) = ctx.compile_expr(expr)?;
            let converted = ctx.emit_convert(val, &expr_ty, &local_ty);
            if let Some((slot, _)) = ctx.stack_slots.get(name) {
                let slot = *slot;
                let addr = ctx.builder.stack_addr(slot);
                ctx.builder.store(addr, converted);
            } else {
                let var = *ctx.locals.get(name).expect("undefined variable in assign");
                ctx.builder.def_var(var, converted);
            }
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
        Stmt::DerefAssign { addr_expr, value } => {
            // The parser stores the full `*expr` as addr_expr (including the Deref).
            // We need the pointer address, so unwrap the outer Deref.
            let ptr_expr = match addr_expr {
                Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    expr: inner,
                } => inner.as_ref(),
                other => other,
            };
            let (addr_val, addr_ty) = ctx.compile_expr(ptr_expr)?;
            let pointee = addr_ty.pointee().clone();
            let (val, val_ty) = ctx.compile_expr(value)?;
            let converted = ctx.emit_convert(val, &val_ty, &pointee);
            ctx.builder.store(addr_val, converted);
        }
        Stmt::IndexAssign { base, index, value } => {
            let (base_val, base_ty) = ctx.compile_expr(base)?;
            let pointee = base_ty.pointee().clone();
            if pointee == CType::Void {
                return Err(TinyErr {
                    line: 0,
                    col: 0,
                    msg: "pointer arithmetic on void* is not allowed".into(),
                });
            }
            let elem_size = base_ty.pointee_size() as i64;
            let (idx_val, idx_ty) = ctx.compile_expr(index)?;
            let idx_val = ctx.emit_convert(idx_val, &idx_ty, &CType::Long);
            let scale = ctx.builder.iconst(elem_size, Type::I64);
            let offset = ctx.builder.mul(idx_val, scale);
            let addr = ctx.builder.add(base_val, offset);
            let (val, val_ty) = ctx.compile_expr(value)?;
            let converted = ctx.emit_convert(val, &val_ty, &pointee);
            ctx.builder.store(addr, converted);
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
