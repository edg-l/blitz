use std::collections::{HashMap, HashSet};

use blitz::ir::builder::{FunctionBuilder, Value, Variable};
use blitz::ir::condcode::CondCode;
use blitz::ir::effectful::BlockId;
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
    pub globals: Vec<blitz::emit::object::GlobalInfo>,
    pub rodata: Vec<blitz::emit::object::GlobalInfo>,
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
        let mut globals = Vec::new();
        let mut rodata = Vec::new();
        let mut string_counter: usize = 0;
        let mut global_types: HashMap<String, CType> = HashMap::new();

        // Process global variable declarations.
        if let Some(ref global_vars) = program.global_vars {
            for gvar in global_vars {
                // Check for duplicate global names.
                if fn_signatures.contains_key(&gvar.name) {
                    return Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!(
                            "global variable '{}' conflicts with function name",
                            gvar.name
                        ),
                    });
                }
                if global_types.contains_key(&gvar.name) {
                    return Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("duplicate global variable '{}'", gvar.name),
                    });
                }

                let (size, align) = compute_global_size_align(&gvar.ty, &struct_registry)?;
                let init = gvar.init.map(|val| {
                    let mut bytes = vec![0u8; size];
                    let le = val.to_le_bytes();
                    let copy_len = size.min(8);
                    bytes[..copy_len].copy_from_slice(&le[..copy_len]);
                    bytes
                });

                globals.push(blitz::emit::object::GlobalInfo {
                    name: gvar.name.clone(),
                    size,
                    align,
                    init,
                });
                global_types.insert(gvar.name.clone(), gvar.ty.clone());
            }
        }

        for func in &program.functions {
            functions.push(compile_fn(
                func,
                &fn_signatures,
                &struct_registry,
                &global_types,
                &mut rodata,
                &mut string_counter,
            )?);
        }
        Ok(Codegen {
            functions,
            globals,
            rodata,
        })
    }
}

struct LoopContext {
    header_block: BlockId,
    exit_block: BlockId,
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
    global_types: &'b HashMap<String, CType>,
    loop_stack: Vec<LoopContext>,
    rodata: &'b mut Vec<blitz::emit::object::GlobalInfo>,
    string_counter: &'b mut usize,
}

impl<'b> FnCtx<'b> {
    fn new(
        builder: &'b mut FunctionBuilder,
        fn_return_type: CType,
        fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
        addressed_vars: HashSet<String>,
        struct_registry: &'b StructRegistry,
        global_types: &'b HashMap<String, CType>,
        rodata: &'b mut Vec<blitz::emit::object::GlobalInfo>,
        string_counter: &'b mut usize,
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
            global_types,
            loop_stack: Vec::new(),
            rodata,
            string_counter,
        }
    }

    /// Look up the C type for a variable, checking stack-allocated, SSA, and global variables.
    fn local_type(&self, name: &str) -> CType {
        if let Some((_, ty)) = self.stack_slots.get(name) {
            ty.clone()
        } else if let Some(ty) = self.local_types.get(name) {
            ty.clone()
        } else if let Some(ty) = self.global_types.get(name) {
            ty.clone()
        } else {
            panic!("local_type: undefined variable '{name}'");
        }
    }

    /// Returns true if the named variable is a global.
    fn is_global(&self, name: &str) -> bool {
        !self.stack_slots.contains_key(name)
            && !self.locals.contains_key(name)
            && self.global_types.contains_key(name)
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

        // Pointer <-> float: compile error (caught at codegen level).
        if (from.is_pointer() && to.is_float()) || (from.is_float() && to.is_pointer()) {
            panic!("cannot convert between pointer and float types");
        }

        // Float <-> float conversions.
        if from.is_float() && to.is_float() {
            return if *from == CType::Float && *to == CType::Double {
                self.builder.float_ext(val)
            } else {
                // Double -> Float
                self.builder.float_trunc(val)
            };
        }

        // Integer -> float conversion.
        if from.is_integer() && to.is_float() {
            let target = to.to_ir_type().unwrap();
            return self.builder.int_to_float(val, target);
        }

        // Float -> integer conversion.
        if from.is_float() && to.is_integer() {
            let target = to.to_ir_type().unwrap();
            return self.builder.float_to_int(val, target);
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

    /// Emit fcmp+select yielding an I32 0/1 value for float comparisons.
    fn emit_fcmp_val(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let flags = self.builder.fcmp(cc, a, b);
        let one = self.builder.iconst(1, Type::I32);
        let zero = self.builder.iconst(0, Type::I32);
        self.builder.select(flags, one, zero)
    }

    /// Emit pointer +/- integer arithmetic: scale the integer by pointee size.
    fn emit_ptr_arith(
        &mut self,
        ptr: Value,
        ptr_ty: &CType,
        idx: Value,
        idx_ty: &CType,
        is_sub: bool,
    ) -> Result<(Value, CType), TinyErr> {
        if *ptr_ty.pointee() == CType::Void {
            return Err(TinyErr {
                line: 0,
                col: 0,
                msg: "pointer arithmetic on void* is not allowed".into(),
            });
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
    fn emit_float_zero(&mut self, ty: &CType) -> Value {
        if *ty == CType::Float {
            self.builder.fconst_f32(0.0f32)
        } else {
            self.builder.fconst(0.0f64)
        }
    }

    /// Compute the byte size of a pointee type, looking up structs in the registry.
    fn pointee_elem_size(&self, pointee: &CType) -> Result<i64, TinyErr> {
        if pointee.is_array() {
            Ok(type_byte_size(pointee, self.struct_registry) as i64)
        } else if let Some(name) = pointee.struct_name() {
            let layout = self.struct_registry.get(name).ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: format!("unknown struct '{name}'"),
            })?;
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
            let zero = self.emit_float_zero(ty);
            return self.builder.fcmp(CondCode::Ne, val, zero);
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
                    if common.is_float() {
                        // Float comparisons use unsigned condition codes (ucomisd/ucomiss).
                        let cc = match op {
                            BinOp::Eq => CondCode::Eq,
                            BinOp::Ne => CondCode::Ne,
                            BinOp::Lt => CondCode::Ult,
                            BinOp::Gt => CondCode::Ugt,
                            BinOp::Le => CondCode::Ule,
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
                    let (val, ty) = self.compile_expr(expr)?;
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
                    Err(TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("undefined variable '{name}'"),
                    })
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::AddrOf,
                expr: inner,
            } => {
                // &var_name: look up the stack slot for the variable, or global
                if let Expr::Var(name) = inner.as_ref() {
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
                        Err(TinyErr {
                            line: 0,
                            col: 0,
                            msg: format!(
                                "address-of variable '{name}' not in stack slots or globals"
                            ),
                        })
                    }
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
                        if ty.is_float() {
                            let zero = self.emit_float_zero(&ty);
                            Ok((self.builder.fsub(zero, val), ty))
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
                            // Deref pointer-to-array: decay to pointer to element
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
            Expr::Call { name, args } => self.compile_call(name, args),
            Expr::Cast { ty, expr } => {
                let (val, from_ty) = self.compile_expr(expr)?;
                let converted = self.emit_convert(val, &from_ty, ty);
                Ok((converted, ty.clone()))
            }
            Expr::Sizeof(ty) => {
                let size = if ty.is_array() {
                    type_byte_size(ty, self.struct_registry) as i64
                } else if let CType::Struct(name) = ty {
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
        }
    }

    fn compile_binop(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
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
                    self.emit_ptr_arith(l, &lt, r, &rt, false)
                } else if lt.is_integer() && rt.is_pointer() {
                    self.emit_ptr_arith(r, &rt, l, &lt, false)
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
                    self.emit_ptr_arith(l, &lt, r, &rt, true)
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
                            return Err(TinyErr {
                                line: 0,
                                col: 0,
                                msg: "modulo operator '%' not supported on float types (use fmod from libm)".into(),
                            });
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

    fn compile_call(&mut self, name: &str, args: &[Expr]) -> Result<(Value, CType), TinyErr> {
        let (ret_type, param_types) = self
            .fn_signatures
            .get(name)
            .map(|(r, p)| (r.clone(), p.clone()))
            .ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: format!("undefined function '{name}'"),
            })?;

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

        let mut arg_vals: Vec<Value> = Vec::new();
        for (i, arg_expr) in args.iter().enumerate() {
            let param_ty = &param_types[i];
            if let Some(sn) = param_ty.struct_name() {
                let struct_name = sn.to_owned();
                let layout = self
                    .struct_registry
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
        cond: &Expr,
        then_expr: &Expr,
        else_expr: &Expr,
    ) -> Result<(Value, CType), TinyErr> {
        let flags = self.compile_cond(cond)?;
        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();

        self.builder.branch(flags, then_block, else_block, &[], &[]);

        // Then arm
        self.builder.set_block(then_block);
        self.builder.seal_block(then_block);
        let (then_val, then_ty) = self.compile_expr(then_expr)?;
        let then_ir_ty = then_ty.to_ir_type().ok_or_else(|| TinyErr {
            line: 0,
            col: 0,
            msg: "ternary operand must be a scalar type".into(),
        })?;
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

/// Compute the total byte size of a type, recursing through arrays and structs.
fn type_byte_size(ty: &CType, registry: &StructRegistry) -> u32 {
    match ty {
        CType::Array(elem, count) => *count as u32 * type_byte_size(elem, registry),
        CType::Struct(name) => {
            let layout = registry
                .get(name.as_str())
                .unwrap_or_else(|| panic!("unknown struct '{name}' in type_byte_size"));
            layout.byte_size
        }
        CType::Ptr(_) => 8,
        CType::Void => panic!("type_byte_size() called on Void"),
        _ => ty.bit_width() / 8,
    }
}

/// Compute the alignment of a type, recursing through arrays and structs.
fn type_alignment(ty: &CType, registry: &StructRegistry) -> u32 {
    match ty {
        CType::Array(elem, _) => type_alignment(elem, registry),
        CType::Struct(name) => {
            let layout = registry
                .get(name.as_str())
                .unwrap_or_else(|| panic!("unknown struct '{name}' in type_alignment"));
            layout.alignment
        }
        CType::Ptr(_) => 8,
        CType::Void => panic!("type_alignment() called on Void"),
        _ => ty.bit_width() / 8,
    }
}

/// Compute (size, alignment) for a global variable type.
fn compute_global_size_align(
    ty: &CType,
    registry: &StructRegistry,
) -> Result<(usize, usize), TinyErr> {
    match ty {
        CType::Void => Err(TinyErr {
            line: 0,
            col: 0,
            msg: "cannot declare global of type void".into(),
        }),
        CType::Array(_, _) => {
            let size = type_byte_size(ty, registry) as usize;
            let align = type_alignment(ty, registry) as usize;
            Ok((size, align))
        }
        CType::Struct(name) => {
            let layout = registry.get(name).ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: format!("unknown struct '{name}'"),
            })?;
            Ok((layout.byte_size as usize, layout.alignment as usize))
        }
        CType::Ptr(_) => Ok((8, 8)),
        _ => {
            let bytes = (ty.bit_width() / 8) as usize;
            Ok((bytes, bytes))
        }
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
        // Check for direct self-reference (non-pointer), including through arrays
        for (fname, fty) in fields {
            let mut inner = fty;
            while let CType::Array(elem, _) = inner {
                inner = elem;
            }
            if *inner == CType::Struct(name.clone()) {
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
            let align = type_alignment(fty, &registry);
            let size = type_byte_size(fty, &registry);
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
        if fty.is_array() {
            // Copy array field word-by-word (8-byte chunks + trailing)
            let total = type_byte_size(fty, ctx.struct_registry) as usize;
            let mut pos = 0;
            while pos < total {
                let remaining = total - pos;
                let (ty, advance) = if remaining >= 8 {
                    (Type::I64, 8)
                } else if remaining >= 4 {
                    (Type::I32, 4)
                } else if remaining >= 2 {
                    (Type::I16, 2)
                } else {
                    (Type::I8, 1)
                };
                let src_elem = if pos > 0 {
                    let off = ctx.builder.iconst(pos as i64, Type::I64);
                    ctx.builder.add(src_field, off)
                } else {
                    src_field
                };
                let dst_elem = if pos > 0 {
                    let off = ctx.builder.iconst(pos as i64, Type::I64);
                    ctx.builder.add(dst_field, off)
                } else {
                    dst_field
                };
                let val = ctx.builder.load(src_elem, ty);
                ctx.builder.store(dst_elem, val);
                pos += advance;
            }
        } else if let CType::Struct(inner_name) = fty {
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
    global_types: &HashMap<String, CType>,
    rodata: &mut Vec<blitz::emit::object::GlobalInfo>,
    string_counter: &mut usize,
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
        global_types,
        rodata,
        string_counter,
    );

    for ((ty, name), val) in fn_def.params.iter().zip(param_vals.iter()) {
        if let Some(sn) = ty.struct_name() {
            // Struct param: val is a hidden pointer. Create local stack slot and copy.
            let struct_name = sn.to_owned();
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
            if ty.is_array() {
                // Arrays always live on the stack
                let size = type_byte_size(ty, ctx.struct_registry);
                let alignment = type_alignment(ty, ctx.struct_registry);
                let slot = ctx.builder.create_stack_slot(size, alignment);
                ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
                // No initializer support for arrays (rejected by parser)
            } else if let Some(sn) = ty.struct_name() {
                let struct_name = sn.to_owned();
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
            if local_ty.is_array() {
                return Err(TinyErr {
                    line: 0,
                    col: 0,
                    msg: "cannot assign to array variable".into(),
                });
            }
            if let Some(struct_name) = local_ty.struct_name() {
                let struct_name = struct_name.to_owned();
                let (src_addr, _) = ctx.compile_expr(expr)?;
                if ctx.is_global(name) {
                    let dst_addr = ctx.builder.global_addr(name);
                    emit_struct_copy(ctx, dst_addr, src_addr, &struct_name)?;
                } else {
                    let (slot, _) = ctx.stack_slots.get(name).ok_or_else(|| TinyErr {
                        line: 0,
                        col: 0,
                        msg: format!("struct variable '{name}' not in stack slots"),
                    })?;
                    let slot = *slot;
                    let dst_addr = ctx.builder.stack_addr(slot);
                    emit_struct_copy(ctx, dst_addr, src_addr, &struct_name)?;
                }
            } else {
                let (val, expr_ty) = ctx.compile_expr(expr)?;
                let converted = ctx.emit_convert(val, &expr_ty, &local_ty);
                if ctx.is_global(name) {
                    let addr = ctx.builder.global_addr(name);
                    ctx.builder.store(addr, converted);
                } else if let Some((slot, _)) = ctx.stack_slots.get(name) {
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
        Stmt::For { cond, update, body } => {
            compile_for(ctx, cond, update.as_deref(), body)?;
        }
        Stmt::Break => {
            let lc = ctx.loop_stack.last().ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: "'break' outside of loop".into(),
            })?;
            let exit = lc.exit_block;
            ctx.builder.jump(exit, &[]);
        }
        Stmt::Continue => {
            let lc = ctx.loop_stack.last().ok_or_else(|| TinyErr {
                line: 0,
                col: 0,
                msg: "'continue' outside of loop".into(),
            })?;
            let header = lc.header_block;
            ctx.builder.jump(header, &[]);
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
            if pointee.is_array() {
                return Err(TinyErr {
                    line: 0,
                    col: 0,
                    msg: "cannot assign to array via deref".into(),
                });
            }
            if let Some(inner_name) = pointee.struct_name() {
                let inner_name = inner_name.to_owned();
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
            if field_ty.is_array() {
                return Err(TinyErr {
                    line: 0,
                    col: 0,
                    msg: "cannot assign directly to array field".into(),
                });
            }
            if let Some(inner_name) = field_ty.struct_name() {
                let inner_name = inner_name.to_owned();
                let (src_addr, _) = ctx.compile_expr(value)?;
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
            let elem_size = ctx.pointee_elem_size(&pointee)?;
            let (idx_val, idx_ty) = ctx.compile_expr(index)?;
            let idx_val = ctx.emit_convert(idx_val, &idx_ty, &CType::Long);
            let scale = ctx.builder.iconst(elem_size, Type::I64);
            let offset = ctx.builder.mul(idx_val, scale);
            let addr = ctx.builder.add(base_val, offset);
            if let Some(inner_name) = pointee.struct_name() {
                let inner_name = inner_name.to_owned();
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
    ctx.loop_stack.push(LoopContext {
        header_block,
        exit_block,
    });
    let body_result = compile_stmts(ctx, body);
    ctx.loop_stack.pop();
    body_result?;
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

fn compile_for(
    ctx: &mut FnCtx,
    cond: &Expr,
    update: Option<&Stmt>,
    body: &[Stmt],
) -> Result<(), TinyErr> {
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let latch_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    // Jump to header from current block.
    ctx.builder.jump(header_block, &[]);

    // Header: do NOT seal yet (back edge from latch not yet known).
    ctx.builder.set_block(header_block);
    let flags = ctx.compile_cond(cond)?;
    ctx.builder.branch(flags, body_block, exit_block, &[], &[]);

    // Body -- continue jumps to latch (where update runs), break jumps to exit.
    ctx.builder.set_block(body_block);
    ctx.builder.seal_block(body_block);
    ctx.loop_stack.push(LoopContext {
        header_block: latch_block, // continue -> latch (runs update before header)
        exit_block,
    });
    let body_result = compile_stmts(ctx, body);
    ctx.loop_stack.pop();
    body_result?;
    if !ctx.is_terminated() {
        ctx.builder.jump(latch_block, &[]);
    }

    // Latch: execute update, then jump back to header.
    ctx.builder.seal_block(latch_block);
    ctx.builder.set_block(latch_block);
    if let Some(upd) = update {
        compile_stmt(ctx, upd)?;
    }
    ctx.builder.jump(header_block, &[]);

    // Now all predecessors of header are known (entry + latch).
    ctx.builder.seal_block(header_block);

    // Exit
    ctx.builder.seal_block(exit_block);
    ctx.builder.set_block(exit_block);

    Ok(())
}
