use std::collections::HashMap;

use blitz::ir::builder::FunctionBuilder;
use blitz::ir::function::Function;
use blitz::ir::types::Type;

use crate::addr_analysis::find_addressed_vars;
use crate::ast::{CType, Expr, FnDef, SpannedExpr, Stmt, UnaryOp};
use crate::error::TinyErr;

use super::structs::{
    StructRegistry, emit_struct_copy, resolve_field, type_alignment, type_byte_size,
};
use super::{FnCtx, LoopContext, err};

pub(super) fn compile_fn(
    fn_def: &FnDef,
    fn_sigs: &HashMap<String, (CType, Vec<CType>)>,
    struct_registry: &StructRegistry,
    global_types: &HashMap<String, CType>,
    rodata: &mut Vec<blitz::emit::object::GlobalInfo>,
    string_counter: &mut usize,
    string_dedup: &mut HashMap<Vec<u8>, String>,
) -> Result<Function, TinyErr> {
    let param_ir_types: Vec<Type> = fn_def
        .params
        .iter()
        .map(|(ty, _)| {
            if ty.is_struct() {
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
        string_dedup,
    );

    for ((ty, name), val) in fn_def.params.iter().zip(param_vals.iter()) {
        if let Some(sn) = ty.struct_name() {
            let struct_name = sn.to_owned();
            let layout = struct_registry
                .get(&struct_name)
                .ok_or_else(|| err(fn_def.span, format!("unknown struct '{struct_name}'")))?;
            let slot = ctx
                .builder
                .create_stack_slot(layout.byte_size, layout.alignment);
            let dst_addr = ctx.builder.stack_addr(slot);
            let src_addr = *val;
            emit_struct_copy(&mut ctx, dst_addr, src_addr, &struct_name, fn_def.span)?;
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

    if !ctx.is_terminated() {
        if fn_def.return_type == CType::Void {
            ctx.builder.ret(None);
        } else {
            let ir_ty = fn_def.return_type.to_ir_type().unwrap();
            let zero = ctx.builder.iconst(0, ir_ty);
            ctx.builder.ret(Some(zero));
        }
    }

    let mut func = builder
        .finalize()
        .map_err(|e| err(fn_def.span, e.to_string()))?;
    func.noinline = fn_def.noinline;
    Ok(func)
}

pub(super) fn compile_stmts(ctx: &mut FnCtx, stmts: &[Stmt]) -> Result<(), TinyErr> {
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
        Stmt::Return(opt_expr, _span) => match opt_expr {
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
        Stmt::ExprStmt(expr, _span) => {
            ctx.compile_expr(expr)?;
        }
        Stmt::VarDecl {
            ty,
            name,
            init,
            span,
            ..
        } => {
            if ty.is_array() {
                let size = type_byte_size(ty, ctx.struct_registry);
                let alignment = type_alignment(ty, ctx.struct_registry);
                let slot = ctx.builder.create_stack_slot(size, alignment);
                ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
            } else if let Some(sn) = ty.struct_name() {
                let struct_name = sn.to_owned();
                let layout = ctx
                    .struct_registry
                    .get(&struct_name)
                    .ok_or_else(|| err(*span, format!("unknown struct '{struct_name}'")))?;
                let byte_size = layout.byte_size;
                let alignment = layout.alignment;
                let slot = ctx.builder.create_stack_slot(byte_size, alignment);
                ctx.stack_slots.insert(name.clone(), (slot, ty.clone()));
                if let Some(init) = init {
                    let (src_addr, _) = ctx.compile_expr(init)?;
                    let dst_addr = ctx.builder.stack_addr(slot);
                    emit_struct_copy(ctx, dst_addr, src_addr, &struct_name, *span)?;
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
        Stmt::Assign {
            name, expr, span, ..
        } => {
            let local_ty = ctx.local_type(name);
            if local_ty.is_array() {
                return Err(err(*span, "cannot assign to array variable"));
            }
            if let Some(struct_name) = local_ty.struct_name() {
                let struct_name = struct_name.to_owned();
                let (src_addr, _) = ctx.compile_expr(expr)?;
                if ctx.is_global(name) {
                    let dst_addr = ctx.builder.global_addr(name);
                    emit_struct_copy(ctx, dst_addr, src_addr, &struct_name, *span)?;
                } else {
                    let (slot, _) = ctx.stack_slots.get(name).ok_or_else(|| {
                        err(
                            *span,
                            format!("struct variable '{name}' not in stack slots"),
                        )
                    })?;
                    let slot = *slot;
                    let dst_addr = ctx.builder.stack_addr(slot);
                    emit_struct_copy(ctx, dst_addr, src_addr, &struct_name, *span)?;
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
            ..
        } => {
            compile_if(ctx, cond, then_body, else_body.as_deref())?;
        }
        Stmt::While { cond, body, .. } => {
            compile_while(ctx, cond, body)?;
        }
        Stmt::For {
            cond, update, body, ..
        } => {
            compile_for(ctx, cond, update.as_deref(), body)?;
        }
        Stmt::DoWhile { body, cond, .. } => {
            compile_do_while(ctx, body, cond)?;
        }
        Stmt::Break(span) => {
            let lc = ctx
                .loop_stack
                .last()
                .ok_or_else(|| err(*span, "'break' outside of loop"))?;
            let exit = lc.exit_block;
            ctx.builder.jump(exit, &[]);
        }
        Stmt::Continue(span) => {
            let lc = ctx
                .loop_stack
                .last()
                .ok_or_else(|| err(*span, "'continue' outside of loop"))?;
            let header = lc.header_block;
            ctx.builder.jump(header, &[]);
        }
        Stmt::DerefAssign {
            addr_expr,
            value,
            span,
            ..
        } => {
            let ptr_expr = match &addr_expr.expr {
                Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    expr: inner,
                } => inner.as_ref(),
                _ => addr_expr,
            };
            let (addr_val, addr_ty) = ctx.compile_expr(ptr_expr)?;
            let pointee = addr_ty.pointee().clone();
            if pointee.is_array() {
                return Err(err(*span, "cannot assign to array via deref"));
            }
            if let Some(inner_name) = pointee.struct_name() {
                let inner_name = inner_name.to_owned();
                let (src_addr, _) = ctx.compile_expr(value)?;
                emit_struct_copy(ctx, addr_val, src_addr, &inner_name, *span)?;
            } else {
                let (val, val_ty) = ctx.compile_expr(value)?;
                let converted = ctx.emit_convert(val, &val_ty, &pointee);
                ctx.builder.store(addr_val, converted);
            }
        }
        Stmt::FieldAssign {
            expr,
            field,
            value,
            span,
            ..
        } => {
            let (struct_addr, struct_ty) = ctx.compile_expr(expr)?;
            let struct_name = match &struct_ty {
                CType::Struct(name) => name.clone(),
                other => {
                    return Err(err(
                        *span,
                        format!("field assign on non-struct type {:?}", other),
                    ));
                }
            };
            let (offset, field_ty) =
                resolve_field(ctx.struct_registry, &struct_name, field, *span)?;
            let offset_val = ctx.builder.iconst(offset as i64, Type::I64);
            let field_addr = ctx.builder.add(struct_addr, offset_val);
            if field_ty.is_array() {
                return Err(err(*span, "cannot assign directly to array field"));
            }
            if let Some(inner_name) = field_ty.struct_name() {
                let inner_name = inner_name.to_owned();
                let (src_addr, _) = ctx.compile_expr(value)?;
                emit_struct_copy(ctx, field_addr, src_addr, &inner_name, *span)?;
            } else {
                let (val, val_ty) = ctx.compile_expr(value)?;
                let converted = ctx.emit_convert(val, &val_ty, &field_ty);
                ctx.builder.store(field_addr, converted);
            }
        }
        Stmt::IndexAssign {
            base,
            index,
            value,
            span,
            ..
        } => {
            let (base_val, base_ty) = ctx.compile_expr(base)?;
            let pointee = base_ty.pointee().clone();
            if pointee == CType::Void {
                return Err(err(*span, "pointer arithmetic on void* is not allowed"));
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
                emit_struct_copy(ctx, addr, src_addr, &inner_name, *span)?;
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
    cond: &SpannedExpr,
    then_body: &[Stmt],
    else_body: Option<&[Stmt]>,
) -> Result<(), TinyErr> {
    let flags = ctx.compile_cond(cond)?;
    let then_block = ctx.builder.create_block();
    let else_block = ctx.builder.create_block();
    ctx.builder.branch(flags, then_block, else_block, &[], &[]);

    ctx.builder.set_block(then_block);
    ctx.builder.seal_block(then_block);
    compile_stmts(ctx, then_body)?;
    let then_terminated = ctx.is_terminated();
    let then_exit = ctx.builder.current_block();

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

fn compile_while(ctx: &mut FnCtx, cond: &SpannedExpr, body: &[Stmt]) -> Result<(), TinyErr> {
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    ctx.builder.jump(header_block, &[]);
    ctx.builder.set_block(header_block);
    let flags = ctx.compile_cond(cond)?;
    ctx.builder.branch(flags, body_block, exit_block, &[], &[]);

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

    ctx.builder.seal_block(header_block);
    ctx.builder.seal_block(exit_block);
    ctx.builder.set_block(exit_block);
    Ok(())
}

fn compile_for(
    ctx: &mut FnCtx,
    cond: &SpannedExpr,
    update: Option<&Stmt>,
    body: &[Stmt],
) -> Result<(), TinyErr> {
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let latch_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    ctx.builder.jump(header_block, &[]);
    ctx.builder.set_block(header_block);
    let flags = ctx.compile_cond(cond)?;
    ctx.builder.branch(flags, body_block, exit_block, &[], &[]);

    ctx.builder.set_block(body_block);
    ctx.builder.seal_block(body_block);
    ctx.loop_stack.push(LoopContext {
        header_block: latch_block,
        exit_block,
    });
    let body_result = compile_stmts(ctx, body);
    ctx.loop_stack.pop();
    body_result?;
    if !ctx.is_terminated() {
        ctx.builder.jump(latch_block, &[]);
    }

    ctx.builder.seal_block(latch_block);
    ctx.builder.set_block(latch_block);
    if let Some(upd) = update {
        compile_stmt(ctx, upd)?;
    }
    ctx.builder.jump(header_block, &[]);

    ctx.builder.seal_block(header_block);
    ctx.builder.seal_block(exit_block);
    ctx.builder.set_block(exit_block);
    Ok(())
}

fn compile_do_while(ctx: &mut FnCtx, body: &[Stmt], cond: &SpannedExpr) -> Result<(), TinyErr> {
    // Desugar do { body } while (cond) as:
    //   while (1) { body; if (!cond) break; }
    // This reuses the standard while-loop structure which handles SSA correctly.
    let header_block = ctx.builder.create_block();
    let body_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();

    // Header: unconditional jump to body (infinite loop condition).
    ctx.builder.jump(header_block, &[]);
    ctx.builder.set_block(header_block);
    ctx.builder.jump(body_block, &[]);

    // Body: execute statements, then check condition at the end.
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
        // Evaluate condition: if true, loop back; if false, exit.
        let flags = ctx.compile_cond(cond)?;
        ctx.builder
            .branch(flags, header_block, exit_block, &[], &[]);
    }

    ctx.builder.seal_block(header_block);
    ctx.builder.seal_block(exit_block);
    ctx.builder.set_block(exit_block);
    Ok(())
}
