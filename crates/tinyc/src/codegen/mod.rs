use std::collections::{HashMap, HashSet};

use blitz::ir::builder::{FunctionBuilder, Value, Variable};
use blitz::ir::condcode::CondCode;
use blitz::ir::effectful::BlockId;
use blitz::ir::function::{Function, StackSlot};
use blitz::ir::types::Type;

use crate::ast::{CType, Program};
use crate::error::TinyErr;
use crate::lexer::Span;

pub(super) fn err(span: Span, msg: impl Into<String>) -> TinyErr {
    TinyErr {
        line: span.line,
        col: span.col,
        msg: msg.into(),
    }
}

mod structs;
use structs::{build_struct_registry, compute_global_size_align, StructRegistry};
mod stmts;
use stmts::compile_fn;
mod expr;

pub struct Codegen {
    pub functions: Vec<Function>,
    pub globals: Vec<blitz::emit::object::GlobalInfo>,
    pub rodata: Vec<blitz::emit::object::GlobalInfo>,
    pub extern_globals: Vec<String>,
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
                    return Err(err(
                        ext.span,
                        format!(
                            "extern function '{}' cannot have struct parameters",
                            ext.name
                        ),
                    ));
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
                return Err(err(
                    func.span,
                    format!("function '{}' cannot have struct return type", func.name),
                ));
            }
            let param_types: Vec<CType> = func.params.iter().map(|(ty, _)| ty.clone()).collect();
            fn_signatures.insert(func.name.clone(), (func.return_type.clone(), param_types));
        }

        let mut functions = Vec::new();
        let mut globals = Vec::new();
        let mut rodata = Vec::new();
        let mut string_counter: usize = 0;
        let mut string_dedup: HashMap<Vec<u8>, String> = HashMap::new();
        let mut global_types: HashMap<String, CType> = HashMap::new();

        // Process global variable declarations.
        for gvar in &program.global_vars {
            // Check for duplicate global names.
            if fn_signatures.contains_key(&gvar.name) {
                return Err(err(
                    gvar.span,
                    format!(
                        "global variable '{}' conflicts with function name",
                        gvar.name
                    ),
                ));
            }
            if global_types.contains_key(&gvar.name) {
                return Err(err(
                    gvar.span,
                    format!("duplicate global variable '{}'", gvar.name),
                ));
            }

            let (size, align) = compute_global_size_align(&gvar.ty, &struct_registry, gvar.span)?;
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

        // Process extern global variable declarations.
        // These are added to global_types (so codegen can reference them)
        // but NOT to globals (no storage allocation -- defined in another file).
        let mut extern_global_names = Vec::new();
        for ext_gvar in &program.extern_globals {
            if global_types.contains_key(&ext_gvar.name) {
                return Err(err(
                    ext_gvar.span,
                    format!("duplicate global variable '{}'", ext_gvar.name),
                ));
            }
            global_types.insert(ext_gvar.name.clone(), ext_gvar.ty.clone());
            extern_global_names.push(ext_gvar.name.clone());
        }

        for func in &program.functions {
            functions.push(compile_fn(
                func,
                &fn_signatures,
                &struct_registry,
                &global_types,
                &mut rodata,
                &mut string_counter,
                &mut string_dedup,
            )?);
        }
        Ok(Codegen {
            functions,
            globals,
            rodata,
            extern_globals: extern_global_names,
        })
    }
}

pub(super) struct LoopContext {
    pub(super) header_block: BlockId,
    pub(super) exit_block: BlockId,
}

pub(super) struct FnCtx<'b> {
    pub(super) builder: &'b mut FunctionBuilder,
    pub(super) locals: HashMap<String, Variable>,
    pub(super) local_types: HashMap<String, CType>,
    pub(super) stack_slots: HashMap<String, (StackSlot, CType)>,
    pub(super) addressed_vars: HashSet<String>,
    pub(super) fn_return_type: CType,
    pub(super) fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
    pub(super) struct_registry: &'b StructRegistry,
    pub(super) global_types: &'b HashMap<String, CType>,
    pub(super) loop_stack: Vec<LoopContext>,
    pub(super) rodata: &'b mut Vec<blitz::emit::object::GlobalInfo>,
    pub(super) string_counter: &'b mut usize,
    pub(super) string_dedup: &'b mut HashMap<Vec<u8>, String>,
}

impl<'b> FnCtx<'b> {
    pub(super) fn new(
        builder: &'b mut FunctionBuilder,
        fn_return_type: CType,
        fn_signatures: &'b HashMap<String, (CType, Vec<CType>)>,
        addressed_vars: HashSet<String>,
        struct_registry: &'b StructRegistry,
        global_types: &'b HashMap<String, CType>,
        rodata: &'b mut Vec<blitz::emit::object::GlobalInfo>,
        string_counter: &'b mut usize,
        string_dedup: &'b mut HashMap<Vec<u8>, String>,
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
            string_dedup,
        }
    }

    /// Look up the C type for a variable, checking stack-allocated, SSA, and global variables.
    pub(super) fn local_type(&self, name: &str) -> CType {
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
    pub(super) fn is_global(&self, name: &str) -> bool {
        !self.stack_slots.contains_key(name)
            && !self.locals.contains_key(name)
            && self.global_types.contains_key(name)
    }

    pub(super) fn is_terminated(&self) -> bool {
        self.builder.is_current_block_terminated()
    }

    /// Sign-extend, zero-extend, or truncate `val` from `from` to `to`.
    pub(super) fn emit_convert(&mut self, val: Value, from: &CType, to: &CType) -> Value {
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
    pub(super) fn emit_promote(&mut self, val: Value, ty: &CType) -> (Value, CType) {
        let promoted = ty.promoted();
        if promoted != *ty {
            let val = self.emit_convert(val, ty, &promoted);
            (val, promoted)
        } else {
            (val, promoted)
        }
    }

    /// Promote both operands then convert to their usual arithmetic common type.
    pub(super) fn emit_usual_conversion(
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
    pub(super) fn emit_icmp_val(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let flags = self.builder.icmp(cc, a, b);
        let one = self.builder.iconst(1, Type::I32);
        let zero = self.builder.iconst(0, Type::I32);
        self.builder.select(flags, one, zero)
    }

    /// Emit fcmp+select yielding an I32 0/1 value for float comparisons.
    /// Handles NaN correctly per IEEE 754 using only NaN-safe Ugt/Uge CCs:
    /// - Eq: (a >= b) AND (b >= a) -- both NaN-safe, true iff equal and ordered
    /// - Ne: NOT((a >= b) AND (b >= a)) -- true iff unequal or NaN
    /// - Lt/Le: swap operands and use Ugt/Uge
    /// - Gt/Ge: already NaN-safe with Ugt/Uge
    pub(super) fn emit_fcmp_val(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let one = self.builder.iconst(1, Type::I32);
        let zero = self.builder.iconst(0, Type::I32);
        match cc {
            CondCode::Eq => {
                // ordered-equal: use fcmp_to_int with OrdEq (Setcc path, no Cmov)
                self.builder.fcmp_to_int(CondCode::OrdEq, a, b)
            }
            CondCode::Ne => {
                // unordered-not-equal: use fcmp_to_int with UnordNe (Setcc path, no Cmov)
                self.builder.fcmp_to_int(CondCode::UnordNe, a, b)
            }
            CondCode::Ult => {
                // ordered-less-than: swap operands, use Ugt (JA is NaN-safe)
                let flags = self.builder.fcmp(CondCode::Ugt, b, a);
                self.builder.select(flags, one, zero)
            }
            CondCode::Ule => {
                // ordered-less-or-equal: swap operands, use Uge (JAE is NaN-safe)
                let flags = self.builder.fcmp(CondCode::Uge, b, a);
                self.builder.select(flags, one, zero)
            }
            _ => {
                // Ugt (JA) and Uge (JAE) are already NaN-safe
                let flags = self.builder.fcmp(cc, a, b);
                self.builder.select(flags, one, zero)
            }
        }
    }
}
