use crate::egraph::egraph::{EGraph, snapshot_all};
use crate::ir::op::Op;
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

/// Propagate known-bits information through the e-graph.
/// Computes output known bits from children's known bits for supported ops.
/// Returns true if any class's known bits were refined.
pub fn propagate_known_bits(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let result = match &snap.op {
            // Bitwise AND: bit is 1 only if both are 1, bit is 0 if either is 0
            Op::And if snap.children.len() == 2 => {
                let a = egraph.get_known_bits(snap.children[0]);
                let b = egraph.get_known_bits(snap.children[1]);
                Some(KnownBits {
                    known_ones: a.known_ones & b.known_ones,
                    known_zeros: a.known_zeros | b.known_zeros,
                })
            }
            // Bitwise OR: bit is 1 if either is 1, bit is 0 only if both are 0
            Op::Or if snap.children.len() == 2 => {
                let a = egraph.get_known_bits(snap.children[0]);
                let b = egraph.get_known_bits(snap.children[1]);
                Some(KnownBits {
                    known_ones: a.known_ones | b.known_ones,
                    known_zeros: a.known_zeros & b.known_zeros,
                })
            }
            // Bitwise XOR: bit is 1 if inputs differ, bit is 0 if inputs are the same
            Op::Xor if snap.children.len() == 2 => {
                let a = egraph.get_known_bits(snap.children[0]);
                let b = egraph.get_known_bits(snap.children[1]);
                Some(KnownBits {
                    known_ones: (a.known_ones & b.known_zeros) | (a.known_zeros & b.known_ones),
                    known_zeros: (a.known_ones & b.known_ones) | (a.known_zeros & b.known_zeros),
                })
            }
            // Shift left by constant: shift known bits left, low bits become known-zero
            Op::Shl if snap.children.len() == 2 => {
                let a = egraph.get_known_bits(snap.children[0]);
                if let Some((shift_amt, _)) = egraph.get_constant(snap.children[1]) {
                    let n = shift_amt as u32;
                    let ty = &egraph
                        .class(egraph.unionfind.find_immutable(snap.class_id))
                        .ty;
                    let mask = type_mask(ty);
                    if n < 64 {
                        let low_mask = if n == 0 { 0 } else { (1u64 << n) - 1 };
                        Some(KnownBits {
                            known_ones: (a.known_ones << n) & mask,
                            known_zeros: ((a.known_zeros << n) | low_mask) & mask,
                        })
                    } else {
                        // Shift by >= 64 is all zeros
                        Some(KnownBits {
                            known_ones: 0,
                            known_zeros: mask,
                        })
                    }
                } else {
                    None
                }
            }
            // Logical shift right by constant: shift right, high bits become known-zero
            Op::Shr if snap.children.len() == 2 => {
                let a = egraph.get_known_bits(snap.children[0]);
                if let Some((shift_amt, _)) = egraph.get_constant(snap.children[1]) {
                    let n = shift_amt as u32;
                    let ty = &egraph
                        .class(egraph.unionfind.find_immutable(snap.class_id))
                        .ty;
                    let mask = type_mask(ty);
                    let width = ty.bit_width();
                    if n < width {
                        let high_mask = mask & !(mask >> n);
                        Some(KnownBits {
                            known_ones: (a.known_ones >> n) & mask,
                            known_zeros: ((a.known_zeros >> n) | high_mask) & mask,
                        })
                    } else {
                        Some(KnownBits {
                            known_ones: 0,
                            known_zeros: mask,
                        })
                    }
                } else {
                    None
                }
            }
            // Arithmetic shift right by constant: high bits replicate sign bit
            Op::Sar if snap.children.len() == 2 => {
                let a = egraph.get_known_bits(snap.children[0]);
                if let Some((shift_amt, _)) = egraph.get_constant(snap.children[1]) {
                    let n = shift_amt as u32;
                    let ty = &egraph
                        .class(egraph.unionfind.find_immutable(snap.class_id))
                        .ty;
                    let mask = type_mask(ty);
                    let width = ty.bit_width();
                    if n < width {
                        let sign_bit = 1u64 << (width - 1);
                        let high_mask = mask & !(mask >> n);
                        let mut ones = (a.known_ones >> n) & mask;
                        let mut zeros = (a.known_zeros >> n) & mask;
                        // If sign bit is known-one, high bits are all ones
                        if a.known_ones & sign_bit != 0 {
                            ones |= high_mask;
                        }
                        // If sign bit is known-zero, high bits are all zeros
                        if a.known_zeros & sign_bit != 0 {
                            zeros |= high_mask;
                        }
                        Some(KnownBits {
                            known_ones: ones,
                            known_zeros: zeros,
                        })
                    } else {
                        // Shift by >= width: result is all sign bits
                        let sign_bit = 1u64 << (width - 1);
                        if a.known_ones & sign_bit != 0 {
                            Some(KnownBits {
                                known_ones: mask,
                                known_zeros: 0,
                            })
                        } else if a.known_zeros & sign_bit != 0 {
                            Some(KnownBits {
                                known_ones: 0,
                                known_zeros: mask,
                            })
                        } else {
                            None
                        }
                    }
                } else {
                    None
                }
            }
            // Zero-extend: upper bits are known-zero
            Op::Zext(target_ty) if snap.children.len() == 1 => {
                let a = egraph.get_known_bits(snap.children[0]);
                let child_canon = egraph.unionfind.find_immutable(snap.children[0]);
                let child_ty = &egraph.class(child_canon).ty;
                if child_ty.is_integer() && target_ty.is_integer() {
                    let child_mask = type_mask(child_ty);
                    let target_mask = type_mask(target_ty);
                    let upper_bits = target_mask & !child_mask;
                    Some(KnownBits {
                        known_ones: a.known_ones & child_mask,
                        known_zeros: (a.known_zeros & child_mask) | upper_bits,
                    })
                } else {
                    None
                }
            }
            // Sign-extend: upper bits replicate sign bit
            Op::Sext(target_ty) if snap.children.len() == 1 => {
                let a = egraph.get_known_bits(snap.children[0]);
                let child_canon = egraph.unionfind.find_immutable(snap.children[0]);
                let child_ty = &egraph.class(child_canon).ty;
                if child_ty.is_integer() && target_ty.is_integer() {
                    let child_mask = type_mask(child_ty);
                    let target_mask = type_mask(target_ty);
                    let upper_bits = target_mask & !child_mask;
                    let sign_bit = 1u64 << (child_ty.bit_width() - 1);
                    let mut ones = a.known_ones & child_mask;
                    let mut zeros = a.known_zeros & child_mask;
                    if a.known_ones & sign_bit != 0 {
                        ones |= upper_bits;
                    }
                    if a.known_zeros & sign_bit != 0 {
                        zeros |= upper_bits;
                    }
                    Some(KnownBits {
                        known_ones: ones,
                        known_zeros: zeros,
                    })
                } else {
                    None
                }
            }
            // Truncate: keep only the lower bits
            Op::Trunc(target_ty) if snap.children.len() == 1 => {
                let a = egraph.get_known_bits(snap.children[0]);
                if target_ty.is_integer() {
                    let target_mask = type_mask(target_ty);
                    Some(KnownBits {
                        known_ones: a.known_ones & target_mask,
                        known_zeros: a.known_zeros & target_mask,
                    })
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(new_kb) = result {
            let canon = egraph.unionfind.find_immutable(snap.class_id);
            let old_kb = egraph.classes[canon.0 as usize].known_bits;
            // Refine: a bit is known if it was already known OR newly computed as known.
            // This is a union of knowledge (not intersection like merge).
            let refined = KnownBits {
                known_zeros: old_kb.known_zeros | new_kb.known_zeros,
                known_ones: old_kb.known_ones | new_kb.known_ones,
            };
            // Check invariant: no bit can be both known-zero and known-one
            debug_assert!(
                refined.known_zeros & refined.known_ones == 0,
                "known-bits conflict in class {:?}: zeros={:#x}, ones={:#x}",
                canon,
                refined.known_zeros,
                refined.known_ones
            );
            if refined != old_kb {
                egraph.classes[canon.0 as usize].known_bits = refined;
                changed = true;
            }
        }
    }
    changed
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

    #[test]
    fn propagate_and_with_constant_mask() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let param = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let mask = g.add(ENode {
            op: Op::Iconst(0xFF, Type::I64),
            children: smallvec![],
        });
        let and = g.add(ENode {
            op: Op::And,
            children: smallvec![param, mask],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(and);
        // Upper 56 bits should be known-zero (from the mask constant having those bits as 0)
        assert_eq!(kb.known_zeros & !0xFFu64, !0xFFu64);
        // Lower 8 bits: param is unknown, mask is all-ones, so And result lower 8 bits are unknown
        // (we can't know them without knowing param)
    }

    #[test]
    fn propagate_or_known_ones() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let param = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let val = g.add(ENode {
            op: Op::Iconst(0xF0, Type::I64),
            children: smallvec![],
        });
        let or = g.add(ENode {
            op: Op::Or,
            children: smallvec![param, val],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(or);
        // Bits 4-7 should be known-one (from the constant)
        assert_eq!(kb.known_ones & 0xF0, 0xF0);
    }

    #[test]
    fn propagate_xor_fully_known() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Iconst(0xFF, Type::I64),
            children: smallvec![],
        });
        let b = g.add(ENode {
            op: Op::Iconst(0x0F, Type::I64),
            children: smallvec![],
        });
        let xor = g.add(ENode {
            op: Op::Xor,
            children: smallvec![a, b],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(xor);
        // 0xFF ^ 0x0F = 0xF0
        assert_eq!(kb.known_ones, 0xF0);
        assert_eq!(kb.known_zeros, !0xF0u64);
    }

    #[test]
    fn propagate_shl_constant_shift() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let shift = g.add(ENode {
            op: Op::Iconst(3, Type::I64),
            children: smallvec![],
        });
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![val, shift],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(shl);
        // 1 << 3 = 8 = 0b1000
        assert_eq!(kb.known_ones, 8);
        // Bits 0-2 should be known-zero from the shift
        assert!(kb.known_zeros & 0b111 == 0b111);
    }

    #[test]
    fn propagate_zext_upper_bits_zero() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let param = g.add(ENode {
            op: Op::Param(0, Type::I8),
            children: smallvec![],
        });
        let zext = g.add(ENode {
            op: Op::Zext(Type::I64),
            children: smallvec![param],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(zext);
        // Upper 56 bits should be known-zero
        assert_eq!(kb.known_zeros & !0xFFu64, !0xFFu64);
        // Lower 8 bits: param is unknown, so unknown
        assert_eq!(kb.known_ones, 0);
    }

    #[test]
    fn propagate_trunc_preserves_lower() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(0x1FF, Type::I32),
            children: smallvec![],
        });
        let trunc = g.add(ENode {
            op: Op::Trunc(Type::I8),
            children: smallvec![val],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(trunc);
        // 0x1FF truncated to I8 = 0xFF: all 8 lower bits are known-one
        assert_eq!(kb.known_ones, 0xFF);
        assert_eq!(kb.known_zeros, 0);
    }

    #[test]
    fn propagate_sext_sign_bit_known_zero() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        // Iconst(42, I8): 42 = 0x2A, sign bit (bit 7) is 0
        let val = g.add(ENode {
            op: Op::Iconst(42, Type::I8),
            children: smallvec![],
        });
        let sext = g.add(ENode {
            op: Op::Sext(Type::I32),
            children: smallvec![val],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(sext);
        // Sign bit of I8 (bit 7) is known-zero, so upper bits 8-31 are known-zero
        let upper_mask = 0xFFFF_FF00u64;
        assert_eq!(kb.known_zeros & upper_mask, upper_mask);
        assert_eq!(kb.known_ones, 42); // lower 8 bits: 42 = 0b00101010
    }

    #[test]
    fn propagate_sext_sign_bit_known_one() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        // Iconst(-1, I8): 0xFF, sign bit (bit 7) is 1
        let val = g.add(ENode {
            op: Op::Iconst(-1, Type::I8),
            children: smallvec![],
        });
        let sext = g.add(ENode {
            op: Op::Sext(Type::I32),
            children: smallvec![val],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(sext);
        // Sign bit is known-one, so upper bits 8-31 are known-one
        let upper_mask = 0xFFFF_FF00u64;
        assert_eq!(kb.known_ones & upper_mask, upper_mask);
        // Lower 8 bits: all ones
        assert_eq!(kb.known_ones & 0xFF, 0xFF);
    }
}
