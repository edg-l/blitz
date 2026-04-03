use blitz::ir::types::Type;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CType {
    Void,
    Char,   // i8, signed
    Short,  // i16
    Int,    // i32
    Long,   // i64
    UChar,  // u8
    UShort, // u16
    UInt,   // u32
    ULong,  // u64
    Float,  // f32
    Double, // f64
    Ptr(Box<CType>),
    Struct(String),
    Array(Box<CType>, usize),
}

impl CType {
    pub fn to_ir_type(&self) -> Option<Type> {
        match self {
            CType::Void => None,
            CType::Char | CType::UChar => Some(Type::I8),
            CType::Short | CType::UShort => Some(Type::I16),
            CType::Int | CType::UInt => Some(Type::I32),
            CType::Long | CType::ULong => Some(Type::I64),
            CType::Float => Some(Type::F32),
            CType::Double => Some(Type::F64),
            CType::Ptr(_) => Some(Type::I64),
            CType::Struct(_) => None,
            CType::Array(_, _) => None,
        }
    }

    pub fn bit_width(&self) -> u32 {
        match self {
            CType::Void => panic!("bit_width called on Void"),
            CType::Char | CType::UChar => 8,
            CType::Short | CType::UShort => 16,
            CType::Int | CType::UInt | CType::Float => 32,
            CType::Long | CType::ULong | CType::Double | CType::Ptr(_) => 64,
            CType::Struct(_) => panic!("use StructRegistry::byte_size()"),
            CType::Array(_, _) => panic!("use array_total_byte_size()"),
        }
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            CType::UChar | CType::UShort | CType::UInt | CType::ULong | CType::Ptr(_)
        )
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, CType::Char | CType::Short | CType::Int | CType::Long)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, CType::Float | CType::Double)
    }

    pub fn is_integer(&self) -> bool {
        !matches!(
            self,
            CType::Void
                | CType::Float
                | CType::Double
                | CType::Ptr(_)
                | CType::Struct(_)
                | CType::Array(_, _)
        )
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, CType::Ptr(_))
    }

    pub fn is_struct(&self) -> bool {
        matches!(self, CType::Struct(_))
    }

    pub fn struct_name(&self) -> Option<&str> {
        match self {
            CType::Struct(name) => Some(name),
            _ => None,
        }
    }

    pub fn is_array(&self) -> bool {
        matches!(self, CType::Array(_, _))
    }

    pub fn array_element(&self) -> &CType {
        match self {
            CType::Array(elem, _) => elem,
            _ => panic!("array_element() called on non-array type {:?}", self),
        }
    }

    pub fn array_len(&self) -> usize {
        match self {
            CType::Array(_, count) => *count,
            _ => panic!("array_len() called on non-array type {:?}", self),
        }
    }

    /// Array-to-pointer decay: Array(elem, _) -> Ptr(elem), non-array unchanged.
    pub fn decay(&self) -> CType {
        match self {
            CType::Array(elem, _) => CType::Ptr(elem.clone()),
            other => other.clone(),
        }
    }

    pub fn pointee(&self) -> &CType {
        match self {
            CType::Ptr(inner) => inner,
            _ => panic!("pointee() called on non-pointer type {:?}", self),
        }
    }

    pub fn pointee_size(&self) -> u32 {
        match self {
            CType::Ptr(inner) => {
                if *inner.as_ref() == CType::Void {
                    panic!("pointee_size() called on void*");
                }
                if inner.is_struct() {
                    panic!("pointee_size() on struct pointer: use StructRegistry");
                }
                if inner.is_array() {
                    panic!("pointee_size() on array pointer: use array_total_byte_size");
                }
                inner.bit_width() / 8
            }
            _ => panic!("pointee_size() called on non-pointer type {:?}", self),
        }
    }

    /// Integer conversion rank per C standard.
    pub fn rank(&self) -> u8 {
        match self {
            CType::Void => 0,
            CType::Char | CType::UChar => 1,
            CType::Short | CType::UShort => 2,
            CType::Int | CType::UInt => 3,
            CType::Long | CType::ULong => 4,
            CType::Ptr(_) => 5,
            CType::Float => panic!("rank() called on float type"),
            CType::Double => panic!("rank() called on double type"),
            CType::Struct(_) => panic!("rank() called on struct type"),
            CType::Array(_, _) => panic!("rank() called on array type"),
        }
    }

    /// Integer promotion per C standard 6.3.1.1: Char/Short/UChar/UShort -> Int.
    /// Float/Double pass through unchanged.
    pub fn promoted(&self) -> CType {
        match self {
            CType::Char | CType::Short | CType::UChar | CType::UShort => CType::Int,
            CType::Struct(_) => panic!("promoted() called on struct type"),
            CType::Array(_, _) => panic!("promoted() called on array type"),
            other => other.clone(),
        }
    }

    /// Usual arithmetic conversion per C standard 6.3.1.8.
    /// Both inputs should already be promoted.
    pub fn usual_arithmetic_conversion(a: &CType, b: &CType) -> CType {
        // If either operand is a pointer, return the pointer type unchanged.
        if a.is_pointer() {
            return a.clone();
        }
        if b.is_pointer() {
            return b.clone();
        }

        // Float promotions take priority: double > float > integer types.
        if matches!(a, CType::Double) || matches!(b, CType::Double) {
            return CType::Double;
        }
        if matches!(a, CType::Float) || matches!(b, CType::Float) {
            return CType::Float;
        }

        if a == b {
            return a.clone();
        }
        if a.is_unsigned() == b.is_unsigned() {
            // Same signedness: return higher rank.
            if a.rank() >= b.rank() {
                a.clone()
            } else {
                b.clone()
            }
        } else {
            let (unsigned, signed) = if a.is_unsigned() { (a, b) } else { (b, a) };
            if unsigned.rank() >= signed.rank() {
                unsigned.clone()
            } else if signed.bit_width() > unsigned.bit_width() {
                // Signed type can represent all values of unsigned type.
                signed.clone()
            } else {
                // Return unsigned version of the signed type.
                match signed {
                    CType::Int => CType::UInt,
                    CType::Long => CType::ULong,
                    // Char/Short are already handled by promotion before this call.
                    _ => unreachable!(),
                }
            }
        }
    }
}

pub struct ExternDecl {
    pub name: String,
    pub return_type: CType,
    pub params: Vec<CType>,
}

pub struct GlobalVar {
    pub name: String,
    pub ty: CType,
    pub init: Option<i64>,
}

pub struct Program {
    pub functions: Vec<FnDef>,
    pub extern_decls: Vec<ExternDecl>,
    pub struct_defs: Vec<(String, Vec<(String, CType)>)>,
    pub global_vars: Option<Vec<GlobalVar>>,
}

pub struct FnDef {
    pub name: String,
    pub return_type: CType,
    pub params: Vec<(CType, String)>,
    pub body: Vec<Stmt>,
    pub noinline: bool,
}

pub enum Stmt {
    Return(Option<Expr>),
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    For {
        cond: Expr,
        update: Option<Box<Stmt>>,
        body: Vec<Stmt>,
    },
    VarDecl {
        ty: CType,
        name: String,
        init: Option<Expr>,
    },
    Assign {
        name: String,
        expr: Expr,
    },
    DerefAssign {
        addr_expr: Expr,
        value: Expr,
    },
    IndexAssign {
        base: Expr,
        index: Expr,
        value: Expr,
    },
    FieldAssign {
        expr: Expr,
        field: String,
        value: Expr,
    },
    Break,
    Continue,
    ExprStmt(Expr),
}

pub enum Expr {
    IntLit(i64),
    /// Float literal: (bits as u64, has_f_suffix). If has_f_suffix, type is float; otherwise double.
    FloatLit(u64, bool),
    StringLit(Vec<u8>),
    Var(String),
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Cast {
        ty: CType,
        expr: Box<Expr>,
    },
    Sizeof(CType),
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    FieldAccess {
        expr: Box<Expr>,
        field: String,
    },
    Ternary {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    Deref,
    AddrOf,
}
