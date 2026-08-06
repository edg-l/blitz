/// IR type system for the Blitz compiler backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Flags,
    Pair(Box<Type>, Box<Type>),
}

impl Type {
    /// Returns the bit width of this type. Panics on `Flags` and `Pair`.
    pub fn bit_width(&self) -> u32 {
        match self {
            Type::I8 => 8,
            Type::I16 => 16,
            Type::I32 => 32,
            Type::I64 => 64,
            Type::F32 => 32,
            Type::F64 => 64,
            Type::Flags => panic!("bit_width() called on Flags type"),
            Type::Pair(_, _) => panic!("bit_width() called on Pair type"),
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    /// Byte size of the type, or `None` for `Flags` and `Pair`.
    ///
    /// The natural size a frontend lays a field out with; see
    /// [`Self::align_bytes`] for the matching alignment and
    /// [`crate::ir::layout::Layout`] for putting the two together.
    pub fn byte_size(&self) -> Option<usize> {
        match self {
            Type::I8 => Some(1),
            Type::I16 => Some(2),
            Type::I32 => Some(4),
            Type::I64 => Some(8),
            Type::F32 => Some(4),
            Type::F64 => Some(8),
            Type::Flags | Type::Pair(_, _) => None,
        }
    }

    /// Natural alignment in bytes, or `None` for `Flags` and `Pair`.
    ///
    /// Every scalar this backend has is naturally aligned to its own size on
    /// x86-64, so this equals [`Self::byte_size`]. It exists separately because
    /// a layout is written in terms of both, and a frontend that says
    /// `align_bytes` where it means alignment keeps saying what it means if that
    /// ever stops being true.
    pub fn align_bytes(&self) -> Option<usize> {
        self.byte_size()
    }
}
