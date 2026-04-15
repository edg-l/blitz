use std::collections::{HashMap, HashSet};

use blitz::ir::builder::Value;
use blitz::ir::types::Type;

use crate::ast::CType;
use crate::error::TinyErr;
use crate::lexer::Span;

use super::{FnCtx, err};

pub(crate) struct StructLayout {
    pub(crate) fields: Vec<(String, CType, u32)>, // (name, type, offset)
    pub(crate) byte_size: u32,
    pub(crate) alignment: u32,
}

pub(super) type StructRegistry = HashMap<String, StructLayout>;

pub(super) fn type_byte_size(ty: &CType, registry: &StructRegistry) -> u32 {
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

pub(super) fn type_alignment(ty: &CType, registry: &StructRegistry) -> u32 {
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

pub(super) fn compute_global_size_align(
    ty: &CType,
    registry: &StructRegistry,
    span: Span,
) -> Result<(usize, usize), TinyErr> {
    match ty {
        CType::Void => Err(err(span, "cannot declare global of type void")),
        CType::Array(_, _) => {
            let size = type_byte_size(ty, registry) as usize;
            let align = type_alignment(ty, registry) as usize;
            Ok((size, align))
        }
        CType::Struct(name) => {
            let layout = registry
                .get(name)
                .ok_or_else(|| err(span, format!("unknown struct '{name}'")))?;
            Ok((layout.byte_size as usize, layout.alignment as usize))
        }
        CType::Ptr(_) => Ok((8, 8)),
        _ => {
            let bytes = (ty.bit_width() / 8) as usize;
            Ok((bytes, bytes))
        }
    }
}

pub(super) fn build_struct_registry(
    struct_defs: &[(String, Vec<(String, CType)>, Span)],
) -> Result<StructRegistry, TinyErr> {
    let mut registry = StructRegistry::new();
    for (name, fields, span) in struct_defs {
        let span = *span;
        if registry.contains_key(name.as_str()) {
            return Err(err(span, format!("duplicate struct definition '{name}'")));
        }
        if fields.is_empty() {
            return Err(err(span, format!("struct '{name}' has no fields")));
        }
        let mut seen = HashSet::new();
        for (fname, _) in fields {
            if !seen.insert(fname.clone()) {
                return Err(err(
                    span,
                    format!("duplicate field '{fname}' in struct '{name}'"),
                ));
            }
        }
        for (fname, fty) in fields {
            let mut inner = fty;
            while let CType::Array(elem, _) = inner {
                inner = elem;
            }
            if *inner == CType::Struct(name.clone()) {
                return Err(err(
                    span,
                    format!(
                        "struct '{name}' has recursive field '{fname}' (use a pointer instead)"
                    ),
                ));
            }
        }

        let mut offset: u32 = 0;
        let mut max_align: u32 = 1;
        let mut layout_fields = Vec::new();

        for (fname, fty) in fields {
            let align = type_alignment(fty, &registry);
            let size = type_byte_size(fty, &registry);
            offset = (offset + align - 1) & !(align - 1);
            layout_fields.push((fname.clone(), fty.clone(), offset));
            offset += size;
            if align > max_align {
                max_align = align;
            }
        }

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

pub(super) fn resolve_field(
    registry: &StructRegistry,
    struct_name: &str,
    field_name: &str,
    span: Span,
) -> Result<(u32, CType), TinyErr> {
    let layout = registry
        .get(struct_name)
        .ok_or_else(|| err(span, format!("unknown struct '{struct_name}'")))?;
    for (fname, fty, foffset) in &layout.fields {
        if fname == field_name {
            return Ok((*foffset, fty.clone()));
        }
    }
    Err(err(
        span,
        format!("struct '{struct_name}' has no field '{field_name}'"),
    ))
}

pub(super) fn emit_struct_copy(
    ctx: &mut FnCtx,
    dst_addr: Value,
    src_addr: Value,
    struct_name: &str,
    span: Span,
) -> Result<(), TinyErr> {
    let layout = ctx.struct_registry.get(struct_name).ok_or_else(|| {
        err(
            span,
            format!("unknown struct '{struct_name}' in emit_struct_copy"),
        )
    })?;
    let fields: Vec<_> = layout.fields.clone();
    for (_, fty, foffset) in &fields {
        let offset_val = ctx.builder.iconst(*foffset as i64, Type::I64);
        let src_field = ctx.builder.add(src_addr, offset_val);
        let dst_field = ctx.builder.add(dst_addr, offset_val);
        if fty.is_array() {
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
            emit_struct_copy(ctx, dst_field, src_field, inner_name, span)?;
        } else {
            let ir_ty = fty.to_ir_type().unwrap();
            let val = ctx.builder.load(src_field, ir_ty);
            ctx.builder.store(dst_field, val);
        }
    }
    Ok(())
}
