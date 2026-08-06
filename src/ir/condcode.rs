/// Condition codes for integer comparisons and conditional x86-64 instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CondCode {
    /// Equal (ZF=1)
    Eq,
    /// Not equal (ZF=0)
    Ne,
    /// Signed less-than (SF≠OF)
    Slt,
    /// Signed less-than-or-equal (ZF=1 or SF≠OF)
    Sle,
    /// Signed greater-than (ZF=0 and SF=OF)
    Sgt,
    /// Signed greater-than-or-equal (SF=OF)
    Sge,
    /// Unsigned less-than (CF=1)
    Ult,
    /// Unsigned less-than-or-equal (CF=1 or ZF=1)
    Ule,
    /// Unsigned greater-than (CF=0 and ZF=0)
    Ugt,
    /// Unsigned greater-than-or-equal (CF=0)
    Uge,
    /// Parity set (PF=1) -- used for NaN detection after ucomisd/ucomiss
    Parity,
    /// Parity clear (PF=0) -- used for "ordered" check after ucomisd/ucomiss
    NotParity,
    /// Ordered equal: ZF=1 AND PF=0. Used for IEEE float == after ucomisd/ucomiss.
    /// Lowered to: sete + setnp + and (or cmove + cmovnp).
    OrdEq,
    /// Unordered not-equal: ZF=0 OR PF=1. Used for IEEE float != after ucomisd/ucomiss.
    /// Lowered to: setne + setp + or (or cmovne + cmovp).
    UnordNe,
}

impl CondCode {
    /// The condition that holds of `(b, a)` exactly when `self` holds of
    /// `(a, b)`.
    ///
    /// Defined for the integer comparison codes only. The parity and composite
    /// float codes describe a flag pattern rather than an ordering, and their
    /// operand order is fixed by the `ucomisd`/`ucomiss` isel rule.
    pub fn swapped(self) -> Option<CondCode> {
        Some(match self {
            CondCode::Eq => CondCode::Eq,
            CondCode::Ne => CondCode::Ne,
            CondCode::Slt => CondCode::Sgt,
            CondCode::Sle => CondCode::Sge,
            CondCode::Sgt => CondCode::Slt,
            CondCode::Sge => CondCode::Sle,
            CondCode::Ult => CondCode::Ugt,
            CondCode::Ule => CondCode::Uge,
            CondCode::Ugt => CondCode::Ult,
            CondCode::Uge => CondCode::Ule,
            CondCode::Parity | CondCode::NotParity | CondCode::OrdEq | CondCode::UnordNe => {
                return None;
            }
        })
    }
}
