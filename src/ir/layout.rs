//! Struct and array layout: offsets, sizes and alignment.
//!
//! Blitz has no aggregate type, and deliberately so: a struct is a block of
//! memory plus a constant offset per field, which is all the backend needs to
//! know to generate good code. What a frontend *does* need is somewhere to work
//! those offsets out, and padding rules belong to the source language rather
//! than to the backend. So this computes a layout from a policy the caller
//! picks, and hands back integers. Nothing here reaches the e-graph, the
//! scheduler or the register allocator.
//!
//! ```
//! use blitzgen::ir::layout::Layout;
//! use blitzgen::ir::types::Type;
//!
//! // struct Point { int x; int y; long tag; }
//! let point = Layout::c(&[Type::I32, Type::I32, Type::I64]);
//! assert_eq!(point.offsets(), [0, 4, 8]);
//! assert_eq!(point.size(), 16);
//! assert_eq!(point.align(), 8);
//!
//! // The same fields with no padding.
//! let packed = Layout::packed(&[Type::I32, Type::I32, Type::I64]);
//! assert_eq!(packed.size(), 16);
//! assert_eq!(packed.align(), 1);
//! ```
//!
//! Pair it with [`FunctionBuilder::field_addr`](crate::ir::builder::FunctionBuilder::field_addr)
//! to address a field without writing the arithmetic out.

use crate::ir::types::Type;

/// One member of a layout: a scalar, or a nested aggregate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Member {
    /// A scalar field, laid out at its own size and alignment.
    Scalar(Type),
    /// A nested struct or array, laid out at its own size and alignment.
    Nested(Layout),
}

impl From<Type> for Member {
    fn from(ty: Type) -> Self {
        Member::Scalar(ty)
    }
}

impl From<Layout> for Member {
    fn from(l: Layout) -> Self {
        Member::Nested(l)
    }
}

impl Member {
    fn size(&self) -> u32 {
        match self {
            Member::Scalar(ty) => ty.byte_size().expect("scalar member has no size") as u32,
            Member::Nested(l) => l.size(),
        }
    }

    fn align(&self) -> u32 {
        match self {
            Member::Scalar(ty) => ty.align_bytes().expect("scalar member has no alignment") as u32,
            Member::Nested(l) => l.align(),
        }
    }
}

/// Where each field of an aggregate sits.
///
/// Build one with [`Layout::c`], [`Layout::packed`], [`Layout::array`] or
/// [`Layout::explicit`], then ask it for [`Layout::offset`] per field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    fields: Vec<Field>,
    size: u32,
    align: u32,
}

/// One field's placement within its [`Layout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Byte offset from the start of the aggregate.
    pub offset: u32,
    /// Byte size of the field.
    pub size: u32,
    /// Byte alignment of the field.
    pub align: u32,
    /// The scalar type to load and store this field with, or `None` when the
    /// field is a nested aggregate and has no single access width.
    pub ty: Option<Type>,
}

fn align_up(offset: u32, align: u32) -> u32 {
    debug_assert!(
        align.is_power_of_two(),
        "alignment {align} is not a power of two"
    );
    (offset + align - 1) & !(align - 1)
}

impl Layout {
    /// C and SysV struct rules: each field at the next offset satisfying its own
    /// alignment, the aggregate aligned to its widest field, and the size
    /// rounded up so that an array of them keeps every element aligned.
    pub fn c(fields: &[Type]) -> Layout {
        Layout::compose(
            &fields
                .iter()
                .cloned()
                .map(Member::Scalar)
                .collect::<Vec<_>>(),
        )
    }

    /// [`Layout::c`] over members, so a struct can contain another struct or an
    /// array.
    pub fn compose(members: &[Member]) -> Layout {
        let mut fields = Vec::with_capacity(members.len());
        let mut offset = 0u32;
        let mut align = 1u32;
        for m in members {
            let (fsize, falign) = (m.size(), m.align());
            offset = align_up(offset, falign);
            align = align.max(falign);
            fields.push(Field {
                offset,
                size: fsize,
                align: falign,
                ty: match m {
                    Member::Scalar(ty) => Some(ty.clone()),
                    Member::Nested(_) => None,
                },
            });
            offset += fsize;
        }
        let size = align_up(offset, align);
        Layout {
            fields,
            size,
            align,
        }
    }

    /// No padding: every field immediately after the previous one, and the
    /// aggregate byte-aligned. The equivalent of `__attribute__((packed))`.
    pub fn packed(fields: &[Type]) -> Layout {
        Layout::compose_packed(
            &fields
                .iter()
                .cloned()
                .map(Member::Scalar)
                .collect::<Vec<_>>(),
        )
    }

    /// [`Layout::packed`] over members.
    pub fn compose_packed(members: &[Member]) -> Layout {
        let mut fields = Vec::with_capacity(members.len());
        let mut offset = 0u32;
        for m in members {
            let fsize = m.size();
            fields.push(Field {
                offset,
                size: fsize,
                align: 1,
                ty: match m {
                    Member::Scalar(ty) => Some(ty.clone()),
                    Member::Nested(_) => None,
                },
            });
            offset += fsize;
        }
        Layout {
            fields,
            size: offset,
            align: 1,
        }
    }

    /// `count` elements of `elem`, each at `elem.stride()`.
    pub fn array(elem: impl Into<Member>, count: u32) -> Layout {
        let elem = elem.into();
        let (esize, ealign) = (elem.size(), elem.align());
        let stride = align_up(esize, ealign);
        let ty = match &elem {
            Member::Scalar(ty) => Some(ty.clone()),
            Member::Nested(_) => None,
        };
        let fields = (0..count)
            .map(|i| Field {
                offset: i * stride,
                size: esize,
                align: ealign,
                ty: ty.clone(),
            })
            .collect();
        Layout {
            fields,
            size: stride * count,
            align: ealign,
        }
    }

    /// A layout whose offsets the caller states outright, for an ABI or a source
    /// language whose rules are not either of the above.
    ///
    /// Nothing is checked beyond the offsets being inside `size`: a frontend
    /// using this has already decided, and second-guessing it here would be the
    /// backend making a layout decision.
    pub fn explicit(fields: &[(Type, u32)], size: u32, align: u32) -> Layout {
        let fields: Vec<Field> = fields
            .iter()
            .map(|(ty, offset)| {
                debug_assert!(
                    offset + ty.byte_size().expect("explicit field has no size") as u32 <= size,
                    "field at offset {offset} does not fit in a layout of {size} bytes"
                );
                Field {
                    offset: *offset,
                    size: ty.byte_size().expect("explicit field has no size") as u32,
                    align: ty.align_bytes().expect("explicit field has no alignment") as u32,
                    ty: Some(ty.clone()),
                }
            })
            .collect();
        Layout {
            fields,
            size,
            align,
        }
    }

    /// A single scalar as a layout, for composing.
    pub fn scalar(ty: Type) -> Layout {
        Layout::compose(&[Member::Scalar(ty)])
    }

    /// Byte offset of field `i`.
    pub fn offset(&self, i: usize) -> u32 {
        self.fields[i].offset
    }

    /// Field `i`.
    pub fn field(&self, i: usize) -> &Field {
        &self.fields[i]
    }

    /// Every field, in declaration order.
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Every field's byte offset, in declaration order.
    pub fn offsets(&self) -> Vec<u32> {
        self.fields.iter().map(|f| f.offset).collect()
    }

    /// Number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the aggregate has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Total size in bytes, including trailing padding.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Alignment in bytes.
    pub fn align(&self) -> u32 {
        self.align
    }

    /// Distance between consecutive elements of an array of this layout.
    pub fn stride(&self) -> u32 {
        align_up(self.size, self.align)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_struct_pads_to_field_alignment() {
        // struct { int x; int y; long tag; } -> 0, 4, 8; size 16, align 8.
        let l = Layout::c(&[Type::I32, Type::I32, Type::I64]);
        assert_eq!(l.offsets(), [0, 4, 8]);
        assert_eq!(l.size(), 16);
        assert_eq!(l.align(), 8);
    }

    #[test]
    fn c_struct_pads_between_and_at_the_end() {
        // struct { char c; long l; char c2; } -> 0, 8, 16; size 24, align 8.
        let l = Layout::c(&[Type::I8, Type::I64, Type::I8]);
        assert_eq!(l.offsets(), [0, 8, 16]);
        assert_eq!(
            l.size(),
            24,
            "trailing padding keeps an array element aligned"
        );
        assert_eq!(l.align(), 8);
    }

    #[test]
    fn packed_has_no_padding() {
        let l = Layout::packed(&[Type::I8, Type::I64, Type::I8]);
        assert_eq!(l.offsets(), [0, 1, 9]);
        assert_eq!(l.size(), 10);
        assert_eq!(l.align(), 1);
    }

    #[test]
    fn array_stride_is_the_padded_size() {
        // An array of the 24-byte struct above.
        let elem = Layout::c(&[Type::I8, Type::I64, Type::I8]);
        assert_eq!(elem.stride(), 24);
        let arr = Layout::array(elem, 3);
        assert_eq!(arr.offsets(), [0, 24, 48]);
        assert_eq!(arr.size(), 72);
        assert_eq!(arr.align(), 8);
    }

    #[test]
    fn scalar_array_strides_by_element_size() {
        let arr = Layout::array(Type::I32, 4);
        assert_eq!(arr.offsets(), [0, 4, 8, 12]);
        assert_eq!(arr.size(), 16);
        assert_eq!(arr.align(), 4);
        assert_eq!(arr.field(2).ty, Some(Type::I32));
    }

    #[test]
    fn nested_struct_aligns_to_its_own_alignment() {
        // struct Inner { int a; long b; }        -> size 16, align 8
        // struct Outer { char c; Inner i; }      -> c at 0, i at 8; size 24
        let inner = Layout::c(&[Type::I32, Type::I64]);
        assert_eq!(inner.size(), 16);
        let outer = Layout::compose(&[Type::I8.into(), inner.into()]);
        assert_eq!(outer.offsets(), [0, 8]);
        assert_eq!(outer.size(), 24);
        assert_eq!(outer.align(), 8);
        assert_eq!(
            outer.field(1).ty,
            None,
            "a nested aggregate has no access width"
        );
    }

    #[test]
    fn explicit_takes_the_offsets_it_is_given() {
        let l = Layout::explicit(&[(Type::I32, 0), (Type::I32, 64)], 128, 64);
        assert_eq!(l.offsets(), [0, 64]);
        assert_eq!(l.size(), 128);
        assert_eq!(l.align(), 64);
    }

    #[test]
    fn empty_struct_is_size_zero_align_one() {
        let l = Layout::c(&[]);
        assert!(l.is_empty());
        assert_eq!(l.size(), 0);
        assert_eq!(l.align(), 1);
    }
}
