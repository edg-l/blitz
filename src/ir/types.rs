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
}
