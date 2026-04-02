use std::collections::{HashMap, HashSet};

use blitz::ir::builder::{FunctionBuilder, Value, Variable};
use blitz::ir::condcode::CondCode;
use blitz::ir::function::{Function, StackSlot};
use blitz::ir::types::Type;

use crate::addr_analysis::find_addressed_vars;
use crate::ast::{BinOp, CType, Expr, FnDef, Program, Stmt, UnaryOp};
use crate::error::TinyErr;

pub struct StructLayout {
    pub fields: Vec<(String, CType, u32)>, // (name, type, offset)
    pub byte_size: u32,
    pub alignment: u32,
}

pub type StructRegistry = HashMap<String, StructLayout>;

pub struct Codegen {
    pub functions: Vec<Function>,
}

impl Codegen {
    pub fn generate(program: &Program) -> Result<Codegen, TinyErr> {
        let struct_registry = build_struct_registry(&program.struct_defs)?;

        // Pre-scan all function signatures before codegen.
        let mut fn_signatures: HashMap<String, (CType, Vec<CType>)> = HashMap::new();

        // Register extern declarations so they are callable with type-checking.
        for ext in &program.extern_decls {
            // Error on struct params in extern declarations
            for param_ty in &ext.params {
                if param_ty.is_struct() {
                    return Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!(
                            "extern function '{}' cannot have struct parameters",
                            ext.name
                        ),
                    });
                }
            }
            fn_signatures.insert(
                ext.name.clone(),
                (ext.return_type.clone(), ext.params.clone()),
            );
        }

        for func in &program.functions {
            // Error on struct return types
            if func.return_type.is_struct() {
                return Err(TinyErr {
                    line: 0,
                    col: 0,
                    msg: format!("function '{}' cannot have struct return type", func.name),
                });
            }
            let param_types: Vec<CType> = func.params.iter().map(|(ty, _)| ty.clone()).collect();
            fn_signatures.insert(func.name.clone(), (func.return_type.clone(), param_types));
        }

        let mut functions = Vec::new();
        for func in &program.functions {
            functions.push(compile_fn(func, &fn_signatures, &struct_registry)?);
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
    struct_registry: &'b StructRegistry,
}

impl<'b> FnCtx<'b> {
    fn new(
        builder: &'b mut FunctionBuilder,
        fn_return_type: CType,
        fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
        addressed_vars: HashSet<String>,
        struct_registry: &'b StructRegistry,
    ) -> Self {
        FnCtx {
            builder,
            locals: HashMap::new(),
            local_types: HashMap::new(),
            stack_slots: HashMap::new(),
            addressed_vars,
            fn_return_type,
            fn_signatures,
            struct_registry,
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
        if from.is_struct() || to.is_struct() {
            panic!("emit_convert() called with struct type");
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
        let elem_size = if ptr_ty.pointee().is_struct() {
            let name = match ptr_ty.pointee() {
                CType::Struct(s) => s.clone(),
                _ => unreachable!(),
            };
            let layout = self.struct_registry.get(&name).ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: format!("unknown struct '{name}'"),
            })?;
            layout.byte_size as i64
        } else {
            ptr_ty.pointee_size() as i64
        };
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
        let elem_size = if ptr_ty.pointee().is_struct() {
            let name = match ptr_ty.pointee() {
                CType::Struct(s) => s.clone(),
                _ => unreachable!(),
            };
            let layout = self.struct_registry.get(&name).ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: format!("unknown struct '{name}'"),
            })?;
            layout.byte_size as i64
        } else {
            ptr_ty.pointee_size() as i64
        };
        let idx = self.emit_convert(idx, idx_ty, &CType::Long);
        let scale = self.builder.iconst(elem_size, Type::I64);
        let offset = self.builder.mul(idx, scale);
        let result = self.builder.sub(ptr, offset);
        Ok((result, ptr_ty.clone()))
    }

    /// Convert a value to Flags using a typed zero constant.
    fn val_to_flags(&mut self, val: Value, ty: &CType) -> Value {
        if ty.is_struct() {
            panic!("val_to_flags() called on struct type");
        }
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
            Expr::StringLit(bytes) => {
                // Append null terminator to get the full data to store.
                let mut data: Vec<u8> = bytes.clone();
                data.push(0);
                // Round up to multiple of 8 so wide stores don't overflow the slot.
                let slot_size = ((data.len() as u32) + 7) & !7;
                let slot = self.builder.create_stack_slot(slot_size, 8);

                // Pack bytes into widest possible stores (I64/I32/I16/I8)
                // to reduce register pressure.
                let base_addr = self.builder.stack_addr(slot);
                let mut pos = 0;
                while pos < data.len() {
                    let remaining = data.len() - pos;
                    let (val, ty, advance) = if remaining >= 8 {
                        let v = i64::from_le_bytes([
                            data[pos],
                            data[pos + 1],
                            data[pos + 2],
                            data[pos + 3],
                            data[pos + 4],
                            data[pos + 5],
                            data[pos + 6],
                            data[pos + 7],
                        ]);
                        (v, Type::I64, 8)
                    } else if remaining >= 4 {
                        let v = u32::from_le_bytes([
                            data[pos],
                            data[pos + 1],
                            data[pos + 2],
                            data[pos + 3],
                        ]) as i64;
                        (v, Type::I32, 4)
                    } else if remaining >= 2 {
                        let v = u16::from_le_bytes([data[pos], data[pos + 1]]) as i64;
                        (v, Type::I16, 2)
                    } else {
                        (data[pos] as i64, Type::I8, 1)
                    };

                    let store_val = self.builder.iconst(val, ty);
                    let addr = if pos > 0 {
                        let offset = self.builder.iconst(pos as i64, Type::I64);
                        self.builder.add(base_addr, offset)
                    } else {
                        base_addr
                    };
                    self.builder.store(addr, store_val);
                    pos += advance;
                }

                Ok((base_addr, CType::Ptr(Box::new(CType::Char))))
            }
            Expr::Var(name) => {
                if let Some((slot, ty)) = self.stack_slots.get(name) {
                    let slot = *slot;
                    let ty = ty.clone();
                    let addr = self.builder.stack_addr(slot);
                    if ty.is_struct() {
                        // Struct: return address, do not load
                        return Ok((addr, ty));
                    }
                    let ir_ty = ty.to_ir_type().unwrap();
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
                } else if let Expr::FieldAccess { expr, field } = inner.as_ref() {
                    // &s.field: compute field address
                    let (struct_addr, struct_ty) = self.compile_expr(expr)?;
                    let struct_name = match &struct_ty {
                        CType::Struct(name) => name.clone(),
                        other => {
                            return Err(TinyErr {
                                line: 0,
                                col: 0,
                                msg: format!("field access on non-struct type {:?}", other),
                            });
                        }
                    };
                    let (offset, field_ty) =
                        resolve_field(self.struct_registry, &struct_name, field)?;
                    let offset_val = self.builder.iconst(offset as i64, Type::I64);
                    let field_addr = self.builder.add(struct_addr, offset_val);
                    Ok((field_addr, CType::Ptr(Box::new(field_ty))))
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
                        if pointee.is_struct() {
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
            Expr::BinOp { op, lhs, rhs } => {
                match op {
                    BinOp::And => {
                        // Short-circuit: if left is false, result is 0.
                        let (l, lt) = self.compile_expr(lhs)?;
                        let (l, lt) = self.emit_promote(l, &lt);
                        let lzero = self.builder.iconst(0, lt.to_ir_type().unwrap());
                        let lcond = self.builder.icmp(CondCode::Ne, l, lzero);

                        let bb_rhs = self.builder.create_block();
                        let (bb_merge, merge_params) =
                            self.builder.create_block_with_params(&[Type::I32]);
                        let merge_val = merge_params[0];

                        let zero_i32 = self.builder.iconst(0, Type::I32);
                        self.builder
                            .branch(lcond, bb_rhs, bb_merge, &[], &[zero_i32]);

                        self.builder.seal_block(bb_rhs);
                        self.builder.set_block(bb_rhs);
                        let (r, rt) = self.compile_expr(rhs)?;
                        let (r, rt) = self.emit_promote(r, &rt);
                        let rzero = self.builder.iconst(0, rt.to_ir_type().unwrap());
                        let rbool = self.emit_icmp_val(CondCode::Ne, r, rzero);
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
                        let (l, lt) = self.emit_promote(l, &lt);
                        let lzero = self.builder.iconst(0, lt.to_ir_type().unwrap());
                        let lcond = self.builder.icmp(CondCode::Ne, l, lzero);

                        let bb_rhs = self.builder.create_block();
                        let (bb_merge, merge_params) =
                            self.builder.create_block_with_params(&[Type::I32]);
                        let merge_val = merge_params[0];

                        let one_i32 = self.builder.iconst(1, Type::I32);
                        self.builder
                            .branch(lcond, bb_merge, bb_rhs, &[one_i32], &[]);

                        self.builder.seal_block(bb_rhs);
                        self.builder.set_block(bb_rhs);
                        let (r, rt) = self.compile_expr(rhs)?;
                        let (r, rt) = self.emit_promote(r, &rt);
                        let rzero = self.builder.iconst(0, rt.to_ir_type().unwrap());
                        let rbool = self.emit_icmp_val(CondCode::Ne, r, rzero);
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

                // Validate arity.
                if args.len() != param_types.len() {
                    return Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!(
                            "function '{}' expects {} argument(s), got {}",
                            name,
                            param_types.len(),
                            args.len()
                        ),
                    });
                }

                // Compile and convert each argument to the expected parameter type.
                let mut arg_vals: Vec<Value> = Vec::new();
                for (i, arg_expr) in args.iter().enumerate() {
                    let param_ty = &param_types[i];
                    if param_ty.is_struct() {
                        // Struct arg: allocate temp slot, copy struct into it, pass address
                        let struct_name = match param_ty {
                            CType::Struct(s) => s.clone(),
                            _ => unreachable!(),
                        };
                        let layout =
                            self.struct_registry
                                .get(&struct_name)
                                .ok_or_else(|| TinyErr {
                                    line: 0,
                                    col: 0,
                                    msg: format!("unknown struct '{struct_name}'"),
                                })?;
                        let byte_size = layout.byte_size;
                        let alignment = layout.alignment;
                        let tmp_slot = self.builder.create_stack_slot(byte_size, alignment);
                        let tmp_addr = self.builder.stack_addr(tmp_slot);
                        let (src_addr, _) = self.compile_expr(arg_expr)?;
                        emit_struct_copy(self, tmp_addr, src_addr, &struct_name)?;
                        arg_vals.push(tmp_addr);
                    } else {
                        let (arg_val, arg_ty) = self.compile_expr(arg_expr)?;
                        let converted = self.emit_convert(arg_val, &arg_ty, param_ty);
                        arg_vals.push(converted);
                    }
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
                let size = if let CType::Struct(name) = ty {
                    let layout = self.struct_registry.get(name).ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("sizeof unknown struct '{name}'"),
                    })?;
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
                    return Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: "pointer arithmetic on void* is not allowed".into(),
                    });
                }
                let elem_size = if pointee.is_struct() {
                    let name = match &pointee {
                        CType::Struct(s) => s.clone(),
                        _ => unreachable!(),
                    };
                    let layout = self.struct_registry.get(&name).ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("unknown struct '{name}'"),
                    })?;
                    layout.byte_size as i64
                } else {
                    base_ty.pointee_size() as i64
                };
                let (idx_val, idx_ty) = self.compile_expr(index)?;
                let idx_val = self.emit_convert(idx_val, &idx_ty, &CType::Long);
                let scale = self.builder.iconst(elem_size, Type::I64);
                let offset = self.builder.mul(idx_val, scale);
                let addr = self.builder.add(base_val, offset);
                if pointee.is_struct() {
                    Ok((addr, pointee))
                } else {
                    let ir_ty = pointee.to_ir_type().unwrap();
                    let loaded = self.builder.load(addr, ir_ty);
                    Ok((loaded, pointee))
                }
            }
            Expr::FieldAccess { expr, field } => {
                let (struct_addr, struct_ty) = self.compile_expr(expr)?;
                let struct_name = match &struct_ty {
                    CType::Struct(name) => name.clone(),
                    other => {
                        return Err(TinyErr {
                            line: 0,
                            col: 0,
                            msg: format!("field access on non-struct type {:?}", other),
                        });
                    }
                };
                let (offset, field_ty) = resolve_field(self.struct_registry, &struct_name, field)?;
                let offset_val = self.builder.iconst(offset as i64, Type::I64);
                let field_addr = self.builder.add(struct_addr, offset_val);
                if field_ty.is_struct() {
                    // Nested struct: return address without loading
                    Ok((field_addr, field_ty))
                } else {
                    let ir_ty = field_ty.to_ir_type().unwrap();
                    let loaded = self.builder.load(field_addr, ir_ty);
                    Ok((loaded, field_ty))
                }
            }
        }
    }
}

fn field_byte_size(ty: &CType, registry: &StructRegistry) -> u32 {
    match ty {
        CType::Struct(name) => {
            let layout = registry
                .get(name)
                .unwrap_or_else(|| panic!("unknown struct '{name}' in field_byte_size"));
            layout.byte_size
        }
        CType::Ptr(_) => 8,
        _ => ty.bit_width() / 8,
    }
}

fn field_alignment(ty: &CType, registry: &StructRegistry) -> u32 {
    match ty {
        CType::Struct(name) => {
            let layout = registry
                .get(name)
                .unwrap_or_else(|| panic!("unknown struct '{name}' in field_alignment"));
            layout.alignment
        }
        CType::Ptr(_) => 8,
        _ => ty.bit_width() / 8,
    }
}

fn build_struct_registry(
    struct_defs: &[(String, Vec<(String, CType)>)],
) -> Result<StructRegistry, TinyErr> {
    let mut registry = StructRegistry::new();
    for (name, fields) in struct_defs {
        if registry.contains_key(name.as_str()) {
            return Err(TinyErr {
                line: 0,
                col: 0,
                msg: format!("duplicate struct definition '{name}'"),
            });
        }
        if fields.is_empty() {
            return Err(TinyErr {
                line: 0,
                col: 0,
                msg: format!("struct '{name}' has no fields"),
            });
        }
        // Check for duplicate field names
        let mut seen = HashSet::new();
        for (fname, _) in fields {
            if !seen.insert(fname.clone()) {
                return Err(TinyErr {
                    line: 0,
                    col: 0,
                    msg: format!("duplicate field '{fname}' in struct '{name}'"),
                });
            }
        }
        // Check for direct self-reference (non-pointer)
        for (fname, fty) in fields {
            if *fty == CType::Struct(name.clone()) {
                return Err(TinyErr {
                    line: 0,
                    col: 0,
                    msg: format!(
                        "struct '{name}' has recursive field '{fname}' (use a pointer instead)"
                    ),
                });
            }
        }

        let mut offset: u32 = 0;
        let mut max_align: u32 = 1;
        let mut layout_fields = Vec::new();

        for (fname, fty) in fields {
            let align = field_alignment(fty, &registry);
            let size = field_byte_size(fty, &registry);
            // Align offset
            offset = (offset + align - 1) & !(align - 1);
            layout_fields.push((fname.clone(), fty.clone(), offset));
            offset += size;
            if align > max_align {
                max_align = align;
            }
        }

        // Pad struct size to alignment
        let byte_size = (offset + max_align - 1) & !(max_align - 1);

        registry.insert(
            name.clone(),
            StructLayout {
                fields: layout_fields,
                byte_size,
                alignment: max_align,
            },
        );
    }
    Ok(registry)
}

fn resolve_field(
    registry: &StructRegistry,
    struct_name: &str,
    field_name: &str,
) -> Result<(u32, CType), TinyErr> {
    let layout = registry.get(struct_name).ok_or_else(|| TinyErr {
        line: 0,
        col: 0,
        msg: format!("unknown struct '{struct_name}'"),
    })?;
    for (fname, fty, foffset) in &layout.fields {
        if fname == field_name {
            return Ok((*foffset, fty.clone()));
        }
    }
    Err(TinyErr {
        line: 0,
        col: 0,
        msg: format!("struct '{struct_name}' has no field '{field_name}'"),
    })
}

fn emit_struct_copy(
    ctx: &mut FnCtx,
    dst_addr: Value,
    src_addr: Value,
    struct_name: &str,
) -> Result<(), TinyErr> {
    let layout = ctx
        .struct_registry
        .get(struct_name)
        .ok_or_else(|| TinyErr {
            line: 0,
            col: 0,
            msg: format!("unknown struct '{struct_name}' in emit_struct_copy"),
        })?;
    let fields: Vec<_> = layout.fields.clone();
    for (_, fty, foffset) in &fields {
        let offset_val = ctx.builder.iconst(*foffset as i64, Type::I64);
        let src_field = ctx.builder.add(src_addr, offset_val);
        let dst_field = ctx.builder.add(dst_addr, offset_val);
        if let CType::Struct(inner_name) = fty {
            emit_struct_copy(ctx, dst_field, src_field, inner_name)?;
        } else {
            let ir_ty = fty.to_ir_type().unwrap();
            let val = ctx.builder.load(src_field, ir_ty);
            ctx.builder.store(dst_field, val);
        }
    }
    Ok(())
}

fn compile_fn(
    fn_def: &FnDef,
    fn_sigs: &HashMap<String, (CType, Vec<CType>)>,
    struct_registry: &StructRegistry,
) -> Result<Function, TinyErr> {
    let param_ir_types: Vec<Type> = fn_def
        .params
        .iter()
        .map(|(ty, _)| {
            if ty.is_struct() {
                // By-value struct params are passed as hidden pointers
                Type::I64
            } else {
                ty.to_ir_type().unwrap()
            }
        })
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
        struct_registry,
    );

    for ((ty, name), val) in fn_def.params.iter().zip(param_vals.iter()) {
        if ty.is_struct() {
            // Struct param: val is a hidden pointer. Create local stack slot and copy.
            let struct_name = match ty {
                CType::Struct(s) => s.clone(),
                _ => unreachable!(),
            };
            let layout = struct_registry.get(&struct_name).ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: format!("unknown struct '{struct_name}'"),
            })?;
            let slot = ctx
                .builder
                .create_stack_slot(layout.byte_size, layout.alignment);
            let dst_addr = ctx.builder.stack_addr(slot);
            let src_addr = *val; // hidden pointer
            emit_struct_copy(&mut ctx, dst_addr, src_addr, &struct_name)?;
            ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
        } else if ctx.addressed_vars.contains(name) {
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

    let mut func = builder.finalize().map_err(|e| TinyErr {
        line: 0,
        col: 0,
        msg: e.to_string(),
    })?;
    func.noinline = fn_def.noinline;
    Ok(func)
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
            if ty.is_struct() {
                let struct_name = match ty {
                    CType::Struct(s) => s.clone(),
                    _ => unreachable!(),
                };
                let layout = ctx
                    .struct_registry
                    .get(&struct_name)
                    .ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("unknown struct '{struct_name}'"),
                    })?;
                let byte_size = layout.byte_size;
                let alignment = layout.alignment;
                let slot = ctx.builder.create_stack_slot(byte_size, alignment);
                ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
                if let Some(init) = init {
                    let (src_addr, _) = ctx.compile_expr(init)?;
                    let dst_addr = ctx.builder.stack_addr(slot);
                    emit_struct_copy(ctx, dst_addr, src_addr, &struct_name)?;
                }
            } else if let Some(init) = init {
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
            } else {
                // Declaration without initializer: just allocate.
                if ctx.addressed_vars.contains(name) {
                    let size = ty.bit_width() / 8;
                    let slot = ctx.builder.create_stack_slot(size, 8);
                    ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
                } else {
                    let ir_ty = ty.to_ir_type().unwrap();
                    let var = ctx.builder.declare_var(ir_ty.clone());
                    let zero = ctx.builder.iconst(0, ir_ty);
                    ctx.builder.def_var(var, zero);
                    ctx.locals.insert(name.clone(), var);
                    ctx.local_types.insert(name.clone(), ty.clone());
                }
            }
        }
        Stmt::Assign { name, expr } => {
            let local_ty = ctx.local_type(name);
            if local_ty.is_struct() {
                let struct_name = match &local_ty {
                    CType::Struct(s) => s.clone(),
                    _ => unreachable!(),
                };
                let (src_addr, _) = ctx.compile_expr(expr)?;
                let (slot, _) = ctx.stack_slots.get(name).ok_or_else(|| TinyErr {
                    line: 0,
                    col: 0,
                    msg: format!("struct variable '{name}' not in stack slots"),
                })?;
                let slot = *slot;
                let dst_addr = ctx.builder.stack_addr(slot);
                emit_struct_copy(ctx, dst_addr, src_addr, &struct_name)?;
            } else {
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
            if pointee.is_struct() {
                let inner_name = match &pointee {
                    CType::Struct(s) => s.clone(),
                    _ => unreachable!(),
                };
                let (src_addr, _) = ctx.compile_expr(value)?;
                emit_struct_copy(ctx, addr_val, src_addr, &inner_name)?;
            } else {
                let (val, val_ty) = ctx.compile_expr(value)?;
                let converted = ctx.emit_convert(val, &val_ty, &pointee);
                ctx.builder.store(addr_val, converted);
            }
        }
        Stmt::FieldAssign { expr, field, value } => {
            // Compile the struct expression to get its address
            let (struct_addr, struct_ty) = ctx.compile_expr(expr)?;
            let struct_name = match &struct_ty {
                CType::Struct(name) => name.clone(),
                other => {
                    return Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("field assign on non-struct type {:?}", other),
                    });
                }
            };
            let (offset, field_ty) = resolve_field(ctx.struct_registry, &struct_name, field)?;
            let offset_val = ctx.builder.iconst(offset as i64, Type::I64);
            let field_addr = ctx.builder.add(struct_addr, offset_val);
            if field_ty.is_struct() {
                let (src_addr, _) = ctx.compile_expr(value)?;
                let inner_name = match &field_ty {
                    CType::Struct(s) => s.clone(),
                    _ => unreachable!(),
                };
                emit_struct_copy(ctx, field_addr, src_addr, &inner_name)?;
            } else {
                let (val, val_ty) = ctx.compile_expr(value)?;
                let converted = ctx.emit_convert(val, &val_ty, &field_ty);
                ctx.builder.store(field_addr, converted);
            }
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
            let elem_size = if pointee.is_struct() {
                let name = match &pointee {
                    CType::Struct(s) => s,
                    _ => unreachable!(),
                };
                ctx.struct_registry
                    .get(name.as_str())
                    .ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("unknown struct '{name}'"),
                    })?
                    .byte_size as i64
            } else {
                base_ty.pointee_size() as i64
            };
            let (idx_val, idx_ty) = ctx.compile_expr(index)?;
            let idx_val = ctx.emit_convert(idx_val, &idx_ty, &CType::Long);
            let scale = ctx.builder.iconst(elem_size, Type::I64);
            let offset = ctx.builder.mul(idx_val, scale);
            let addr = ctx.builder.add(base_val, offset);
            if pointee.is_struct() {
                let inner_name = match &pointee {
                    CType::Struct(s) => s.clone(),
                    _ => unreachable!(),
                };
                let (src_addr, _) = ctx.compile_expr(value)?;
                emit_struct_copy(ctx, addr, src_addr, &inner_name)?;
            } else {
                let (val, val_ty) = ctx.compile_expr(value)?;
                let converted = ctx.emit_convert(val, &val_ty, &pointee);
                ctx.builder.store(addr, converted);
            }
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
