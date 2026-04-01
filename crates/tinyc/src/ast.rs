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
    Ptr(Box<CType>),
}

impl CType {
    pub fn to_ir_type(&self) -> Option<Type> {
        match self {
            CType::Void => None,
            CType::Char | CType::UChar => Some(Type::I8),
            CType::Short | CType::UShort => Some(Type::I16),
            CType::Int | CType::UInt => Some(Type::I32),
            CType::Long | CType::ULong => Some(Type::I64),
            CType::Ptr(_) => Some(Type::I64),
        }
    }

    pub fn bit_width(&self) -> u32 {
        match self {
            CType::Void => panic!("bit_width called on Void"),
            CType::Char | CType::UChar => 8,
            CType::Short | CType::UShort => 16,
            CType::Int | CType::UInt => 32,
            CType::Long | CType::ULong | CType::Ptr(_) => 64,
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

    pub fn is_integer(&self) -> bool {
        !matches!(self, CType::Void | CType::Ptr(_))
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, CType::Ptr(_))
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
        }
    }

    /// Integer promotion per C standard 6.3.1.1: Char/Short/UChar/UShort -> Int.
    pub fn promoted(&self) -> CType {
        match self {
            CType::Char | CType::Short | CType::UChar | CType::UShort => CType::Int,
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

pub struct Program {
    pub functions: Vec<FnDef>,
}

pub struct FnDef {
    pub name: String,
    pub return_type: CType,
    pub params: Vec<(CType, String)>,
    pub body: Vec<Stmt>,
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
    VarDecl {
        ty: CType,
        name: String,
        init: Expr,
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
    ExprStmt(Expr),
}

pub enum Expr {
    IntLit(i64),
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
