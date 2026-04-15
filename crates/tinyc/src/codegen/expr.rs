use blitz::ir::builder::Value;
use blitz::ir::condcode::CondCode;
use blitz::ir::types::Type;

use crate::ast::{BinOp, CType, Expr, SpannedExpr, UnaryOp};
use crate::error::TinyErr;
use crate::lexer::Span;

use super::structs::{emit_struct_copy, resolve_field, type_byte_size};
use super::{err, FnCtx};

impl<'b> FnCtx<'b> {
    /// Emit pointer +/- integer arithmetic: scale the integer by pointee size.
    pub(super) fn emit_ptr_arith(
        &mut self,
        ptr: Value,
        ptr_ty: &CType,
        idx: Value,
        span: Span,
        idx_ty: &CType,
        is_sub: bool,
    ) -> Result<(Value, CType), TinyErr> {
        if *ptr_ty.pointee() == CType::Void {
            return Err(err(span, "pointer arithmetic on void* is not allowed"));
        }
        let elem_size = self.pointee_elem_size(ptr_ty.pointee())?;
        let idx = self.emit_convert(idx, idx_ty, &CType::Long);
        let scale = self.builder.iconst(elem_size, Type::I64);
        let offset = self.builder.mul(idx, scale);
        let result = if is_sub {
            self.builder.sub(ptr, offset)
        } else {
            self.builder.add(ptr, offset)
        };
        Ok((result, ptr_ty.clone()))
    }

    /// Build a float zero constant matching the given float type.
    pub(super) fn emit_float_zero(&mut self, ty: &CType) -> Value {
        if *ty == CType::Float {
            self.builder.fconst_f32(0.0f32)
        } else {
            self.builder.fconst(0.0f64)
        }
    }

    /// Compute the byte size of a pointee type, looking up structs in the registry.
    pub(super) fn pointee_elem_size(&self, pointee: &CType) -> Result<i64, TinyErr> {
        let span = Span::default(); // used for structural errors in helpers
        if pointee.is_array() {
            Ok(type_byte_size(pointee, self.struct_registry) as i64)
        } else if let Some(name) = pointee.struct_name() {
            let layout = self
                .struct_registry
                .get(name)
                .ok_or_else(|| err(span, format!("unknown struct '{name}'")))?;
            Ok(layout.byte_size as i64)
        } else {
            Ok((pointee.bit_width() / 8) as i64)
        }
    }

    /// Convert a value to Flags using a typed zero constant.
    fn val_to_flags(&mut self, val: Value, ty: &CType) -> Value {
        if ty.is_struct() {
            panic!("val_to_flags() called on struct type");
        }
        if ty.is_float() {
            // NaN is truthy, so use UnordNe: true if val!=0 or val is NaN.
            let zero = self.emit_float_zero(ty);
            return self.builder.fcmp(CondCode::UnordNe, val, zero);
        }
        let ir_ty = ty.to_ir_type().unwrap();
        let zero = self.builder.iconst(0, ir_ty);
        self.builder.icmp(CondCode::Ne, val, zero)
    }

    /// Compile an expression as a branch condition, returning Flags directly.
    pub(super) fn compile_cond(&mut self, sexpr: &SpannedExpr) -> Result<Value, TinyErr> {
        match &sexpr.expr {
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
                    if common.is_float() {
                        // NaN-aware float comparisons for branch conditions.
                        // Eq/Ne: OrdEq/UnordNe expanded to multi-instruction jumps in terminator.
                        // Lt/Le: swap operands and use Ugt/Uge (JA/JAE are NaN-safe).
                        // Gt/Ge: Ugt/Uge are already NaN-safe.
                        let cc = match op {
                            BinOp::Eq => CondCode::OrdEq,
                            BinOp::Ne => CondCode::UnordNe,
                            BinOp::Lt => return Ok(self.builder.fcmp(CondCode::Ugt, r, l)),
                            BinOp::Le => return Ok(self.builder.fcmp(CondCode::Uge, r, l)),
                            BinOp::Gt => CondCode::Ugt,
                            BinOp::Ge => CondCode::Uge,
                            _ => unreachable!(),
                        };
                        return Ok(self.builder.fcmp(cc, l, r));
                    }
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
                    let (val, ty) = self.compile_expr(sexpr)?;
                    Ok(self.val_to_flags(val, &ty))
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::Not,
                expr: inner,
            } => {
                let (val, ty) = self.compile_expr(inner)?;
                if ty.is_float() {
                    let zero = self.emit_float_zero(&ty);
                    Ok(self.builder.fcmp(CondCode::Eq, val, zero))
                } else {
                    let (val, ty) = self.emit_promote(val, &ty);
                    let zero = self.builder.iconst(0, ty.to_ir_type().unwrap());
                    Ok(self.builder.icmp(CondCode::Eq, val, zero))
                }
            }
            _ => {
                let (val, ty) = self.compile_expr(sexpr)?;
                Ok(self.val_to_flags(val, &ty))
            }
        }
    }

    pub(super) fn compile_expr(&mut self, sexpr: &SpannedExpr) -> Result<(Value, CType), TinyErr> {
        let span = sexpr.span;
        match &sexpr.expr {
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
            Expr::FloatLit(bits, is_float) => {
                if *is_float {
                    // f suffix: interpret as f32
                    let f64_val = f64::from_bits(*bits);
                    let f32_val = f64_val as f32;
                    let val = self.builder.fconst_f32(f32_val);
                    Ok((val, CType::Float))
                } else {
                    // No suffix: double
                    let f64_val = f64::from_bits(*bits);
                    let val = self.builder.fconst(f64_val);
                    Ok((val, CType::Double))
                }
            }
            Expr::StringLit(bytes) => {
                let label = format!(".L.str.{}", self.string_counter);
                *self.string_counter += 1;

                let mut data: Vec<u8> = bytes.clone();
                data.push(0u8);

                self.rodata.push(blitz::emit::object::GlobalInfo {
                    name: label.clone(),
                    size: data.len(),
                    align: 1,
                    init: Some(data),
                });

                let addr = self.builder.global_addr(&label);
                Ok((addr, CType::Ptr(Box::new(CType::Char))))
            }
            Expr::Var(name) => {
                if let Some((slot, ty)) = self.stack_slots.get(name) {
                    let slot = *slot;
                    let ty = ty.clone();
                    let addr = self.builder.stack_addr(slot);
                    if ty.is_array() {
                        // Array: decay to pointer
                        return Ok((addr, ty.decay()));
                    }
                    if ty.is_struct() {
                        // Struct: return address, do not load
                        return Ok((addr, ty));
                    }
                    let ir_ty = ty.to_ir_type().unwrap();
                    let val = self.builder.load(addr, ir_ty);
                    Ok((val, ty))
                } else if let Some(var) = self.locals.get(name) {
                    let var = *var;
                    let ty = self.local_types[name].clone();
                    let val = self.builder.use_var(var);
                    Ok((val, ty))
                } else if let Some(ty) = self.global_types.get(name) {
                    let ty = ty.clone();
                    let addr = self.builder.global_addr(name);
                    if ty.is_array() {
                        return Ok((addr, ty.decay()));
                    }
                    if ty.is_struct() {
                        return Ok((addr, ty));
                    }
                    let ir_ty = ty.to_ir_type().unwrap();
                    let val = self.builder.load(addr, ir_ty);
                    Ok((val, ty))
                } else {
                    Err(err(span, format!("undefined variable '{name}'")))
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => {
                // &var_name: look up the stack slot for the variable, or global
                if let Expr::Var(name) = &inner.expr {
                    if let Some((slot, var_ty)) = self.stack_slots.get(name) {
                        let slot = *slot;
                        let var_ty = var_ty.clone();
                        let addr = self.builder.stack_addr(slot);
                        Ok((addr, CType::Ptr(Box::new(var_ty))))
                    } else if let Some(ty) = self.global_types.get(name) {
                        let ty = ty.clone();
                        let addr = self.builder.global_addr(name);
                        Ok((addr, CType::Ptr(Box::new(ty))))
                    } else {
                        Err(err(
                            span,
                            format!("address-of variable '{name}' not in stack slots or globals"),
                        ))
                    }
                } else if let Expr::FieldAccess { expr, field } = &inner.expr {
                    // &s.field: compute field address
                    let (struct_addr, struct_ty) = self.compile_expr(expr)?;
                    let struct_name = match &struct_ty {
                        CType::Struct(name) => name.clone(),
                        other => {
                            return Err(err(
                                span,
                                format!("field access on non-struct type {:?}", other),
                            ));
                        }
                    };
                    let (offset, field_ty) =
                        resolve_field(self.struct_registry, &struct_name, field, span)?;
                    let offset_val = self.builder.iconst(offset as i64, Type::I64);
                    let field_addr = self.builder.add(struct_addr, offset_val);
                    Ok((field_addr, CType::Ptr(Box::new(field_ty))))
                } else {
                    Err(err(span, "address-of applied to non-variable expression"))
                }
            }
            Expr::UnaryOp { op, expr } => {
                let (val, ty) = self.compile_expr(expr)?;
                match op {
                    UnaryOp::Neg => {
                        if ty.is_float() {
                            // Use -0.0 (not +0.0) so fsub(-0.0, x) is IEEE-correct
                            // for all cases including -(+0.0) = -0.0.
                            let neg_zero = if ty == CType::Float {
                                self.builder.fconst_f32(-0.0f32)
                            } else {
                                self.builder.fconst(-0.0f64)
                            };
                            Ok((self.builder.fsub(neg_zero, val), ty))
                        } else {
                            let (val, ty) = self.emit_promote(val, &ty);
                            Ok((self.builder.neg(val), ty))
                        }
                    }
                    UnaryOp::Not => {
                        // !x == (x == 0) -> I32 result
                        if ty.is_float() {
                            let zero = self.emit_float_zero(&ty);
                            Ok((self.emit_fcmp_val(CondCode::Eq, val, zero), CType::Int))
                        } else {
                            let (val, ty) = self.emit_promote(val, &ty);
                            let zero = self.builder.iconst(0, ty.to_ir_type().unwrap());
                            Ok((self.emit_icmp_val(CondCode::Eq, val, zero), CType::Int))
                        }
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
                        if pointee.is_array() {
                            // Deref pointer-to-array: decay one more level, return address
                            Ok((val, pointee.decay()))
                        } else if pointee.is_struct() {
                            // Struct pointer deref: return address, don't load
                            Ok((val, pointee))
                        } else {
                            let ir_ty = pointee.to_ir_type().unwrap();
                            let loaded = self.builder.load(val, ir_ty);
                            Ok((loaded, pointee))
                        }
                    }
                    UnaryOp::AddrOf => {
                        unreachable!("AddrOf handled by earlier match arm")
                    }
                }
            }
            Expr::BinOp { op, lhs, rhs } => self.compile_binop(*op, lhs, rhs),
            Expr::Call { name, args } => self.compile_call(name, args, span),
            Expr::Cast { ty, expr } => {
                let (val, from_ty) = self.compile_expr(expr)?;
                let converted = self.emit_convert(val, &from_ty, ty);
                Ok((converted, ty.clone()))
            }
            Expr::Sizeof(ty) => {
                if matches!(ty, CType::Void) {
                    return Err(err(span, "sizeof(void) is not allowed"));
                }
                let size = if ty.is_array() {
                    type_byte_size(ty, self.struct_registry) as i64
                } else if let CType::Struct(name) = ty {
                    let layout = self
                        .struct_registry
                        .get(name)
                        .ok_or_else(|| err(span, format!("sizeof unknown struct '{name}'")))?;
                    layout.byte_size as i64
                } else {
                    ty.bit_width() as i64 / 8
                };
                let val = self.builder.iconst(size, Type::I64);
                Ok((val, CType::ULong))
            }
            Expr::Index { base, index } => {
                let (base_val, base_ty) = self.compile_expr(base)?;
                let pointee = base_ty.pointee().clone();
                if pointee == CType::Void {
                    return Err(err(span, "pointer arithmetic on void* is not allowed"));
                }
                let elem_size = self.pointee_elem_size(&pointee)?;
                let (idx_val, idx_ty) = self.compile_expr(index)?;
                let idx_val = self.emit_convert(idx_val, &idx_ty, &CType::Long);
                let scale = self.builder.iconst(elem_size, Type::I64);
                let offset = self.builder.mul(idx_val, scale);
                let addr = self.builder.add(base_val, offset);
                if pointee.is_array() {
                    // Multidimensional: decay one more level, return address
                    Ok((addr, pointee.decay()))
                } else if pointee.is_struct() {
                    Ok((addr, pointee))
                } else {
                    let ir_ty = pointee.to_ir_type().unwrap();
                    let loaded = self.builder.load(addr, ir_ty);
                    Ok((loaded, pointee))
                }
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => self.compile_ternary(cond, then_expr, else_expr),
            Expr::FieldAccess { expr, field } => {
                let (struct_addr, struct_ty) = self.compile_expr(expr)?;
                let struct_name = match &struct_ty {
                    CType::Struct(name) => name.clone(),
                    other => {
                        return Err(err(
                            span,
                            format!("field access on non-struct type {:?}", other),
                        ));
                    }
                };
                let (offset, field_ty) =
                    resolve_field(self.struct_registry, &struct_name, field, span)?;
                let offset_val = self.builder.iconst(offset as i64, Type::I64);
                let field_addr = self.builder.add(struct_addr, offset_val);
                if field_ty.is_array() {
                    // Array field: decay to pointer
                    Ok((field_addr, field_ty.decay()))
                } else if field_ty.is_struct() {
                    // Nested struct: return address without loading
                    Ok((field_addr, field_ty))
                } else {
                    let ir_ty = field_ty.to_ir_type().unwrap();
                    let loaded = self.builder.load(field_addr, ir_ty);
                    Ok((loaded, field_ty))
                }
            }
            Expr::PreIncrement(inner) | Expr::PreDecrement(inner) => {
                let is_inc = matches!(&sexpr.expr, Expr::PreIncrement(_));
                self.compile_inc_dec(inner, is_inc, true, span)
            }
            Expr::PostIncrement(inner) | Expr::PostDecrement(inner) => {
                let is_inc = matches!(&sexpr.expr, Expr::PostIncrement(_));
                self.compile_inc_dec(inner, is_inc, false, span)
            }
            Expr::Comma(left, right) => {
                self.compile_expr(left)?;
                self.compile_expr(right)
            }
        }
    }

    /// Compile pre/post increment/decrement. `is_pre` = true means return new value.
    fn compile_inc_dec(
        &mut self,
        inner: &SpannedExpr,
        is_inc: bool,
        is_pre: bool,
        span: Span,
    ) -> Result<(Value, CType), TinyErr> {
        match &inner.expr {
            Expr::Var(name) => {
                let ty = self.local_type(name);
                if ty.is_struct() || ty.is_array() {
                    return Err(err(span, "cannot increment/decrement struct or array"));
                }
                // Read current value
                let old_val = if self.is_global(name) {
                    let addr = self.builder.global_addr(name);
                    self.builder.load(addr, ty.to_ir_type().unwrap())
                } else if let Some((slot, _)) = self.stack_slots.get(name) {
                    let slot = *slot;
                    let addr = self.builder.stack_addr(slot);
                    self.builder.load(addr, ty.to_ir_type().unwrap())
                } else {
                    let var = *self
                        .locals
                        .get(name)
                        .ok_or_else(|| err(span, format!("undefined variable '{name}'")))?;
                    self.builder.use_var(var)
                };
                // Compute new value
                let new_val = if ty.is_float() {
                    let one = if ty == CType::Float {
                        self.builder.fconst_f32(1.0f32)
                    } else {
                        self.builder.fconst(1.0f64)
                    };
                    if is_inc {
                        self.builder.fadd(old_val, one)
                    } else {
                        self.builder.fsub(old_val, one)
                    }
                } else if ty.is_pointer() {
                    // Pointer increment: advance by pointee size
                    let pointee = ty.pointee();
                    let elem_size = self.pointee_elem_size(pointee)?;
                    let step = self.builder.iconst(elem_size, Type::I64);
                    if is_inc {
                        self.builder.add(old_val, step)
                    } else {
                        self.builder.sub(old_val, step)
                    }
                } else {
                    let ir_ty = ty.to_ir_type().unwrap();
                    let one = self.builder.iconst(1, ir_ty);
                    if is_inc {
                        self.builder.add(old_val, one)
                    } else {
                        self.builder.sub(old_val, one)
                    }
                };
                // Write back
                if self.is_global(name) {
                    let addr = self.builder.global_addr(name);
                    self.builder.store(addr, new_val);
                } else if let Some((slot, _)) = self.stack_slots.get(name) {
                    let slot = *slot;
                    let addr = self.builder.stack_addr(slot);
                    self.builder.store(addr, new_val);
                } else {
                    let var = *self.locals.get(name).unwrap();
                    self.builder.def_var(var, new_val);
                };
                Ok(if is_pre { (new_val, ty) } else { (old_val, ty) })
            }
            Expr::UnaryOp {
                op: UnaryOp::Deref,
                expr: addr_expr,
            } => {
                let (addr, ptr_ty) = self.compile_expr(addr_expr)?;
                let pointee = ptr_ty.pointee().clone();
                let ir_ty = pointee
                    .to_ir_type()
                    .ok_or_else(|| err(span, "cannot increment/decrement void pointer deref"))?;
                let old_val = self.builder.load(addr, ir_ty.clone());
                let new_val = if pointee.is_float() {
                    let one = if pointee == CType::Float {
                        self.builder.fconst_f32(1.0f32)
                    } else {
                        self.builder.fconst(1.0f64)
                    };
                    if is_inc {
                        self.builder.fadd(old_val, one)
                    } else {
                        self.builder.fsub(old_val, one)
                    }
                } else {
                    let one = self.builder.iconst(1, ir_ty);
                    if is_inc {
                        self.builder.add(old_val, one)
                    } else {
                        self.builder.sub(old_val, one)
                    }
                };
                self.builder.store(addr, new_val);
                Ok(if is_pre {
                    (new_val, pointee)
                } else {
                    (old_val, pointee)
                })
            }
            _ => Err(err(
                span,
                "increment/decrement requires a variable or dereference",
            )),
        }
    }

    fn compile_binop(
        &mut self,
        op: BinOp,
        lhs: &SpannedExpr,
        rhs: &SpannedExpr,
    ) -> Result<(Value, CType), TinyErr> {
        match op {
            BinOp::And => {
                // Short-circuit: if left is false, result is 0.
                let (l, lt) = self.compile_expr(lhs)?;
                let lcond = self.val_to_flags(l, &lt);

                let bb_rhs = self.builder.create_block();
                let (bb_merge, merge_params) = self.builder.create_block_with_params(&[Type::I32]);
                let merge_val = merge_params[0];

                let zero_i32 = self.builder.iconst(0, Type::I32);
                self.builder
                    .branch(lcond, bb_rhs, bb_merge, &[], &[zero_i32]);

                self.builder.seal_block(bb_rhs);
                self.builder.set_block(bb_rhs);
                let (r, rt) = self.compile_expr(rhs)?;
                let r_flags = self.val_to_flags(r, &rt);
                let one = self.builder.iconst(1, Type::I32);
                let zero2 = self.builder.iconst(0, Type::I32);
                let rbool = self.builder.select(r_flags, one, zero2);
                if !self.builder.is_current_block_terminated() {
                    self.builder.jump(bb_merge, &[rbool]);
                }

                self.builder.seal_block(bb_merge);
                self.builder.set_block(bb_merge);
                Ok((merge_val, CType::Int))
            }
            BinOp::Or => {
                // Short-circuit: if left is true, result is 1.
                let (l, lt) = self.compile_expr(lhs)?;
                let lcond = self.val_to_flags(l, &lt);

                let bb_rhs = self.builder.create_block();
                let (bb_merge, merge_params) = self.builder.create_block_with_params(&[Type::I32]);
                let merge_val = merge_params[0];

                let one_i32 = self.builder.iconst(1, Type::I32);
                self.builder
                    .branch(lcond, bb_merge, bb_rhs, &[one_i32], &[]);

                self.builder.seal_block(bb_rhs);
                self.builder.set_block(bb_rhs);
                let (r, rt) = self.compile_expr(rhs)?;
                let r_flags = self.val_to_flags(r, &rt);
                let one = self.builder.iconst(1, Type::I32);
                let zero2 = self.builder.iconst(0, Type::I32);
                let rbool = self.builder.select(r_flags, one, zero2);
                if !self.builder.is_current_block_terminated() {
                    self.builder.jump(bb_merge, &[rbool]);
                }

                self.builder.seal_block(bb_merge);
                self.builder.set_block(bb_merge);
                Ok((merge_val, CType::Int))
            }
            // Shift operators: promote independently, result type is promoted left type.
            BinOp::Shl => {
                let (l, lt) = self.compile_expr(lhs)?;
                let (l, lt) = self.emit_promote(l, &lt);
                let (r, _rt) = self.compile_expr(rhs)?;
                let (r, _rt) = self.emit_promote(r, &_rt);
                let r = self.emit_convert(r, &_rt, &lt);
                Ok((self.builder.shl(l, r), lt))
            }
            BinOp::Shr => {
                let (l, lt) = self.compile_expr(lhs)?;
                let (l, lt) = self.emit_promote(l, &lt);
                let (r, _rt) = self.compile_expr(rhs)?;
                let (r, _rt) = self.emit_promote(r, &_rt);
                let r = self.emit_convert(r, &_rt, &lt);
                if lt.is_signed() {
                    Ok((self.builder.sar(l, r), lt))
                } else {
                    Ok((self.builder.shr(l, r), lt))
                }
            }
            BinOp::Add => {
                let (l, lt) = self.compile_expr(lhs)?;
                let (r, rt) = self.compile_expr(rhs)?;
                if lt.is_pointer() && rt.is_integer() {
                    self.emit_ptr_arith(l, &lt, r, lhs.span, &rt, false)
                } else if lt.is_integer() && rt.is_pointer() {
                    self.emit_ptr_arith(r, &rt, l, rhs.span, &lt, false)
                } else {
                    let (l, r, common) = self.emit_usual_conversion(l, &lt, r, &rt);
                    if common.is_float() {
                        Ok((self.builder.fadd(l, r), common))
                    } else {
                        Ok((self.builder.add(l, r), common))
                    }
                }
            }
            BinOp::Sub => {
                let (l, lt) = self.compile_expr(lhs)?;
                let (r, rt) = self.compile_expr(rhs)?;
                if lt.is_pointer() && rt.is_integer() {
                    self.emit_ptr_arith(l, &lt, r, lhs.span, &rt, true)
                } else {
                    let (l, r, common) = self.emit_usual_conversion(l, &lt, r, &rt);
                    if common.is_float() {
                        Ok((self.builder.fsub(l, r), common))
                    } else {
                        Ok((self.builder.sub(l, r), common))
                    }
                }
            }
            _ => {
                let (l, lt) = self.compile_expr(lhs)?;
                let (r, rt) = self.compile_expr(rhs)?;
                let (l, r, common) = self.emit_usual_conversion(l, &lt, r, &rt);
                match op {
                    BinOp::Mul => {
                        if common.is_float() {
                            Ok((self.builder.fmul(l, r), common))
                        } else {
                            Ok((self.builder.mul(l, r), common))
                        }
                    }
                    BinOp::Div => {
                        if common.is_float() {
                            Ok((self.builder.fdiv(l, r), common))
                        } else if common.is_unsigned() {
                            Ok((self.builder.udiv(l, r), common))
                        } else {
                            Ok((self.builder.sdiv(l, r), common))
                        }
                    }
                    BinOp::Mod => {
                        if common.is_float() {
                            return Err(err(
                                lhs.span,
                                "modulo operator '%' not supported on float types (use fmod from libm)",
                            ));
                        }
                        if common.is_unsigned() {
                            Ok((self.builder.urem(l, r), common))
                        } else {
                            Ok((self.builder.srem(l, r), common))
                        }
                    }
                    BinOp::Eq => {
                        if common.is_float() {
                            Ok((self.emit_fcmp_val(CondCode::Eq, l, r), CType::Int))
                        } else {
                            Ok((self.emit_icmp_val(CondCode::Eq, l, r), CType::Int))
                        }
                    }
                    BinOp::Ne => {
                        if common.is_float() {
                            Ok((self.emit_fcmp_val(CondCode::Ne, l, r), CType::Int))
                        } else {
                            Ok((self.emit_icmp_val(CondCode::Ne, l, r), CType::Int))
                        }
                    }
                    BinOp::Lt => {
                        if common.is_float() {
                            Ok((self.emit_fcmp_val(CondCode::Ult, l, r), CType::Int))
                        } else {
                            let cc = if common.is_unsigned() {
                                CondCode::Ult
                            } else {
                                CondCode::Slt
                            };
                            Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                        }
                    }
                    BinOp::Gt => {
                        if common.is_float() {
                            Ok((self.emit_fcmp_val(CondCode::Ugt, l, r), CType::Int))
                        } else {
                            let cc = if common.is_unsigned() {
                                CondCode::Ugt
                            } else {
                                CondCode::Sgt
                            };
                            Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                        }
                    }
                    BinOp::Le => {
                        if common.is_float() {
                            Ok((self.emit_fcmp_val(CondCode::Ule, l, r), CType::Int))
                        } else {
                            let cc = if common.is_unsigned() {
                                CondCode::Ule
                            } else {
                                CondCode::Sle
                            };
                            Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                        }
                    }
                    BinOp::Ge => {
                        if common.is_float() {
                            Ok((self.emit_fcmp_val(CondCode::Uge, l, r), CType::Int))
                        } else {
                            let cc = if common.is_unsigned() {
                                CondCode::Uge
                            } else {
                                CondCode::Sge
                            };
                            Ok((self.emit_icmp_val(cc, l, r), CType::Int))
                        }
                    }
                    BinOp::BitAnd => Ok((self.builder.and(l, r), common)),
                    BinOp::BitOr => Ok((self.builder.or(l, r), common)),
                    BinOp::BitXor => Ok((self.builder.xor(l, r), common)),
                    BinOp::Add | BinOp::Sub | BinOp::And | BinOp::Or | BinOp::Shl | BinOp::Shr => {
                        unreachable!()
                    }
                }
            }
        }
    }

    fn compile_call(
        &mut self,
        name: &str,
        args: &[SpannedExpr],
        call_span: Span,
    ) -> Result<(Value, CType), TinyErr> {
        let (ret_type, param_types) = self
            .fn_signatures
            .get(name)
            .map(|(r, p)| (r.clone(), p.clone()))
            .ok_or_else(|| err(call_span, format!("undefined function '{name}'")))?;

        if args.len() != param_types.len() {
            return Err(err(
                call_span,
                format!(
                    "function '{}' expects {} argument(s), got {}",
                    name,
                    param_types.len(),
                    args.len()
                ),
            ));
        }

        let mut arg_vals: Vec<Value> = Vec::new();
        for (i, arg_expr) in args.iter().enumerate() {
            let param_ty = &param_types[i];
            if let Some(sn) = param_ty.struct_name() {
                let struct_name = sn.to_owned();
                let layout = self
                    .struct_registry
                    .get(&struct_name)
                    .ok_or_else(|| err(call_span, format!("unknown struct '{struct_name}'")))?;
                let byte_size = layout.byte_size;
                let alignment = layout.alignment;
                let tmp_slot = self.builder.create_stack_slot(byte_size, alignment);
                let tmp_addr = self.builder.stack_addr(tmp_slot);
                let (src_addr, _) = self.compile_expr(arg_expr)?;
                emit_struct_copy(self, tmp_addr, src_addr, &struct_name, call_span)?;
                arg_vals.push(tmp_addr);
            } else {
                let (arg_val, arg_ty) = self.compile_expr(arg_expr)?;
                let converted = self.emit_convert(arg_val, &arg_ty, param_ty);
                arg_vals.push(converted);
            }
        }

        let ret_ir_tys: Vec<Type> = ret_type.to_ir_type().map(|t| vec![t]).unwrap_or_default();
        let results = self.builder.call(name, &arg_vals, &ret_ir_tys);

        if ret_type == CType::Void {
            let dummy = self.builder.iconst(0, Type::I32);
            Ok((dummy, CType::Int))
        } else {
            Ok((results[0], ret_type))
        }
    }

    fn compile_ternary(
        &mut self,
        cond: &SpannedExpr,
        then_expr: &SpannedExpr,
        else_expr: &SpannedExpr,
    ) -> Result<(Value, CType), TinyErr> {
        let flags = self.compile_cond(cond)?;
        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();

        self.builder.branch(flags, then_block, else_block, &[], &[]);

        // Then arm
        self.builder.set_block(then_block);
        self.builder.seal_block(then_block);
        let (then_val, then_ty) = self.compile_expr(then_expr)?;
        let then_ir_ty = then_ty
            .to_ir_type()
            .ok_or_else(|| err(then_expr.span, "ternary operand must be a scalar type"))?;
        let then_exit = self.builder.current_block();
        let then_terminated = self.is_terminated();

        // Else arm
        self.builder.set_block(else_block);
        self.builder.seal_block(else_block);
        let (else_val, else_ty) = self.compile_expr(else_expr)?;
        let else_exit = self.builder.current_block();
        let else_terminated = self.is_terminated();

        // Merge
        let (merge_block, merge_params) = self.builder.create_block_with_params(&[then_ir_ty]);
        let result = merge_params[0];

        if !then_terminated {
            self.builder.set_block(then_exit.unwrap());
            self.builder.jump(merge_block, &[then_val]);
        }
        if !else_terminated {
            self.builder.set_block(else_exit.unwrap());
            let converted = self.emit_convert(else_val, &else_ty, &then_ty);
            self.builder.jump(merge_block, &[converted]);
        }

        self.builder.seal_block(merge_block);
        self.builder.set_block(merge_block);
        Ok((result, then_ty.clone()))
    }
}
