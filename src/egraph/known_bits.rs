use crate::ir::types::Type;

/// Tracks which bits of an integer value are provably zero or one.
///
/// Invariant: `(known_zeros & known_ones) == 0` — a bit cannot be both.
/// Bits not in either mask are unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownBits {
    pub known_zeros: u64,
    pub known_ones: u64,
}

impl KnownBits {
    /// No bits are known.
    pub fn unknown() -> Self {
        KnownBits {
            known_zeros: 0,
            known_ones: 0,
        }
    }

    /// All bits are known from a constant value, masked to the type's width.
    pub fn from_constant(val: i64, ty: &Type) -> Self {
        if !ty.is_integer() {
            return Self::unknown();
        }
        let mask = type_mask(ty);
        let v = val as u64 & mask;
        KnownBits {
            known_zeros: !v & mask,
            known_ones: v,
        }
    }

    /// Merge (join) two KnownBits: intersection of knowledge.
    /// A bit is only known if BOTH inputs agree on it.
    pub fn merge(a: &KnownBits, b: &KnownBits) -> KnownBits {
        KnownBits {
            known_zeros: a.known_zeros & b.known_zeros,
            known_ones: a.known_ones & b.known_ones,
        }
    }

    /// Returns true if any bits are known.
    pub fn has_info(&self) -> bool {
        self.known_zeros != 0 || self.known_ones != 0
    }

    /// If all bits of the type are known, return the constant value.
    pub fn is_constant(&self, ty: &Type) -> Option<i64> {
        if !ty.is_integer() {
            return None;
        }
        let mask = type_mask(ty);
        if (self.known_zeros | self.known_ones) & mask == mask {
            // All bits are known. Sign-extend from the type's width.
            let width = ty.bit_width();
            let val = self.known_ones;
            if width < 64 {
                let sign_bit = 1u64 << (width - 1);
                if val & sign_bit != 0 {
                    // Sign-extend
                    Some((val | !mask) as i64)
                } else {
                    Some(val as i64)
                }
            } else {
                Some(val as i64)
            }
        } else {
            None
        }
    }
}

/// Bitmask for the given type's width. Returns u64::MAX for 64-bit types.
pub fn type_mask(ty: &Type) -> u64 {
    if !ty.is_integer() {
        return 0;
    }
    let width = ty.bit_width();
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::Type;

    #[test]
    fn constant_i64_all_bits_known() {
        let kb = KnownBits::from_constant(42, &Type::I64);
        assert_eq!(kb.known_ones, 42);
        assert_eq!(kb.known_zeros, !42u64);
        assert!(kb.has_info());
    }

    #[test]
    fn constant_i8_masked() {
        let kb = KnownBits::from_constant(-1, &Type::I8);
        assert_eq!(kb.known_ones, 0xFF);
        assert_eq!(kb.known_zeros, 0);
    }

    #[test]
    fn constant_i8_value_42() {
        let kb = KnownBits::from_constant(42, &Type::I8);
        assert_eq!(kb.known_ones, 42);
        assert_eq!(kb.known_zeros, 0xFF & !42u64);
    }

    #[test]
    fn unknown_has_no_info() {
        let kb = KnownBits::unknown();
        assert!(!kb.has_info());
        assert_eq!(kb.known_zeros, 0);
        assert_eq!(kb.known_ones, 0);
    }

    #[test]
    fn merge_intersects_knowledge() {
        let a = KnownBits {
            known_zeros: 0,
            known_ones: 0b01,
        }; // bit 0 is 1
        let b = KnownBits {
            known_zeros: 0b10,
            known_ones: 0,
        }; // bit 1 is 0
        let merged = KnownBits::merge(&a, &b);
        // Neither bit is commonly known between both
        assert_eq!(merged.known_zeros, 0);
        assert_eq!(merged.known_ones, 0);
    }

    #[test]
    fn merge_preserves_common_knowledge() {
        let a = KnownBits {
            known_zeros: 0b10,
            known_ones: 0b01,
        };
        let b = KnownBits {
            known_zeros: 0b10,
            known_ones: 0b01,
        };
        let merged = KnownBits::merge(&a, &b);
        assert_eq!(merged.known_zeros, 0b10);
        assert_eq!(merged.known_ones, 0b01);
    }

    #[test]
    fn is_constant_i64() {
        let kb = KnownBits::from_constant(42, &Type::I64);
        assert_eq!(kb.is_constant(&Type::I64), Some(42));
    }

    #[test]
    fn is_constant_i8_negative() {
        let kb = KnownBits::from_constant(-1, &Type::I8);
        // -1 as i8 = 0xFF, sign-extended back to i64 = -1
        assert_eq!(kb.is_constant(&Type::I8), Some(-1));
    }

    #[test]
    fn is_constant_returns_none_when_unknown() {
        let kb = KnownBits::unknown();
        assert_eq!(kb.is_constant(&Type::I64), None);
    }

    #[test]
    fn non_integer_type_is_unknown() {
        let kb = KnownBits::from_constant(42, &Type::F64);
        assert!(!kb.has_info());
    }

    #[test]
    fn type_mask_values() {
        assert_eq!(type_mask(&Type::I8), 0xFF);
        assert_eq!(type_mask(&Type::I16), 0xFFFF);
        assert_eq!(type_mask(&Type::I32), 0xFFFF_FFFF);
        assert_eq!(type_mask(&Type::I64), u64::MAX);
    }

    #[test]
    fn egraph_iconst_has_known_bits() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let id = g.add(ENode {
            op: Op::Iconst(42, Type::I64),
            children: smallvec![],
        });
        let kb = g.get_known_bits(id);
        assert_eq!(kb.known_ones, 42);
        assert_eq!(kb.known_zeros, !42u64);
    }

    #[test]
    fn egraph_param_has_unknown_bits() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let id = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let kb = g.get_known_bits(id);
        assert!(!kb.has_info());
    }
}
