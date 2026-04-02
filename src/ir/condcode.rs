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
}
