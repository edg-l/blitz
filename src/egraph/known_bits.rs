use smallvec::smallvec;

use crate::egraph::egraph::{EGraph, snapshot_all};
use crate::egraph::enode::ENode;
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
                    let ty = &egraph
                        .class(egraph.unionfind.find_immutable(snap.class_id))
                        .ty;
                    let width = ty.bit_width();
                    // Out-of-range shift: x86 uses CL mod 64; don't derive known bits
                    if shift_amt < 0 || shift_amt >= width as i64 {
                        None
                    } else {
                        let n = shift_amt as u32;
                        let mask = type_mask(ty);
                        let low_mask = if n == 0 { 0 } else { (1u64 << n) - 1 };
                        Some(KnownBits {
                            known_ones: (a.known_ones << n) & mask,
                            known_zeros: ((a.known_zeros << n) | low_mask) & mask,
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
                    let ty = &egraph
                        .class(egraph.unionfind.find_immutable(snap.class_id))
                        .ty;
                    let width = ty.bit_width();
                    if shift_amt < 0 || shift_amt >= width as i64 {
                        None
                    } else {
                        let n = shift_amt as u32;
                        let mask = type_mask(ty);
                        let high_mask = mask & !(mask >> n);
                        Some(KnownBits {
                            known_ones: (a.known_ones >> n) & mask,
                            known_zeros: ((a.known_zeros >> n) | high_mask) & mask,
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
                    let ty = &egraph
                        .class(egraph.unionfind.find_immutable(snap.class_id))
                        .ty;
                    let width = ty.bit_width();
                    if shift_amt < 0 || shift_amt >= width as i64 {
                        None
                    } else {
                        let n = shift_amt as u32;
                        let mask = type_mask(ty);
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

/// Apply optimization rules that exploit known-bits information.
/// - Redundant And removal: And(x, mask) = x when mask doesn't clear any possibly-set bits
/// - Known-constant promotion: if all bits of a class are known, add an Iconst node
pub fn apply_known_bits_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        match &snap.op {
            // Redundant And removal: And(x, mask_const) where the mask preserves all
            // possibly-set bits of x (i.e., bits outside the mask are already known-zero in x).
            Op::And if snap.children.len() == 2 => {
                // Try both orderings: And(x, const) and And(const, x)
                let (val_child, const_child) = if egraph.get_constant(snap.children[1]).is_some() {
                    (snap.children[0], snap.children[1])
                } else if egraph.get_constant(snap.children[0]).is_some() {
                    (snap.children[1], snap.children[0])
                } else {
                    continue;
                };

                if let Some((mask_val, _)) = egraph.get_constant(const_child) {
                    let mask = mask_val as u64;
                    let x_kb = egraph.get_known_bits(val_child);
                    let canon = egraph.unionfind.find_immutable(snap.class_id);
                    let ty = &egraph.class(canon).ty;

                    if !ty.is_integer() {
                        continue;
                    }
                    let ty_mask = type_mask(ty);

                    // If all bits outside the mask are known-zero in x,
                    // then And(x, mask) = x (the mask is redundant).
                    // Condition: (~mask & ~x.known_zeros & ty_mask) == 0
                    // i.e., every bit that the mask would clear is already known-zero.
                    let bits_outside_mask = !mask & ty_mask;
                    if bits_outside_mask != 0 && (bits_outside_mask & !x_kb.known_zeros) == 0 {
                        let val_canon = egraph.unionfind.find_immutable(val_child);
                        let and_canon = egraph.unionfind.find_immutable(snap.class_id);
                        if and_canon != val_canon {
                            egraph.merge(snap.class_id, val_child);
                            changed = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Known-constant promotion: if all bits are known, add Iconst and merge.
    // Do this in a separate pass to avoid interfering with the And removal above.
    let snaps2 = snapshot_all(egraph);
    // Collect classes we've already seen to avoid processing the same class multiple times.
    let mut seen = std::collections::HashSet::new();
    for snap in &snaps2 {
        let canon = egraph.unionfind.find_immutable(snap.class_id);
        if !seen.insert(canon) {
            continue;
        }
        let ty = egraph.class(canon).ty.clone();
        if !ty.is_integer() {
            continue;
        }
        // Skip if the class already has a constant value
        if egraph.get_constant(canon).is_some() {
            continue;
        }
        let kb = egraph.classes[canon.0 as usize].known_bits;
        if let Some(val) = kb.is_constant(&ty) {
            let iconst_id = egraph.add(ENode {
                op: Op::Iconst(val, ty),
                children: smallvec![],
            });
            let iconst_canon = egraph.unionfind.find_immutable(iconst_id);
            let class_canon = egraph.unionfind.find_immutable(canon);
            if iconst_canon != class_canon {
                egraph.merge(canon, iconst_id);
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

    #[test]
    fn redundant_and_after_zext() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::egraph::phases::{CompileOptions, run_phases};
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
        let mask = g.add(ENode {
            op: Op::Iconst(0xFF, Type::I64),
            children: smallvec![],
        });
        let and = g.add(ENode {
            op: Op::And,
            children: smallvec![zext, mask],
        });

        // Run full saturation to propagate known bits and apply rules
        let opts = CompileOptions::default();
        run_phases(&mut g, &opts).expect("no blowup");

        // The And should be eliminated: And(Zext(I64, param_i8), 0xFF) = Zext(I64, param_i8)
        // because Zext already zeroed upper bits, and 0xFF keeps all 8 lower bits.
        assert_eq!(
            g.find(and),
            g.find(zext),
            "And(Zext(I64, param_i8), 0xFF) should equal Zext(I64, param_i8)"
        );
    }

    #[test]
    fn known_constant_from_or_chain() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::egraph::phases::{CompileOptions, run_phases};
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        // Build: Shl(Iconst(1), Iconst(3)) -- this is 8
        let one = g.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let three = g.add(ENode {
            op: Op::Iconst(3, Type::I64),
            children: smallvec![],
        });
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![one, three],
        });

        let opts = CompileOptions::default();
        run_phases(&mut g, &opts).expect("no blowup");

        // After propagation + constant promotion, Shl(1, 3) should be in same class as Iconst(8)
        let eight = g.add(ENode {
            op: Op::Iconst(8, Type::I64),
            children: smallvec![],
        });
        assert_eq!(
            g.find(shl),
            g.find(eight),
            "Shl(1, 3) should be promoted to Iconst(8) via known-bits"
        );
    }

    #[test]
    fn dead_mask_elimination() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::egraph::phases::{CompileOptions, run_phases};
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        // And(Shl(1, 4), 0xFF) -- Shl(1,4) = 0x10, And with 0xFF doesn't change it
        let one = g.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let four = g.add(ENode {
            op: Op::Iconst(4, Type::I64),
            children: smallvec![],
        });
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![one, four],
        });
        let mask = g.add(ENode {
            op: Op::Iconst(0xFF, Type::I64),
            children: smallvec![],
        });
        let and = g.add(ENode {
            op: Op::And,
            children: smallvec![shl, mask],
        });

        let opts = CompileOptions::default();
        run_phases(&mut g, &opts).expect("no blowup");

        // Shl(1,4) = 0x10, And(0x10, 0xFF) = 0x10, so And should merge with Shl
        // (or both merge with Iconst(16))
        assert_eq!(
            g.find(and),
            g.find(shl),
            "And(Shl(1,4), 0xFF) should equal Shl(1,4)"
        );
    }

    // ── Shift edge case tests ───────────────────────────────────────────────

    #[test]
    fn propagate_shl_by_zero() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(0xFF, Type::I64),
            children: smallvec![],
        });
        let zero = g.add(ENode {
            op: Op::Iconst(0, Type::I64),
            children: smallvec![],
        });
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![val, zero],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(shl);
        // Shl(0xFF, 0) = 0xFF
        assert_eq!(kb.known_ones, 0xFF);
    }

    #[test]
    fn propagate_shl_negative_amount_skipped() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let neg = g.add(ENode {
            op: Op::Iconst(-1, Type::I64),
            children: smallvec![],
        });
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![val, neg],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(shl);
        // Out-of-range shift: no bits should be derived
        assert!(!kb.has_info());
    }

    #[test]
    fn propagate_shl_overshift_skipped() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let big = g.add(ENode {
            op: Op::Iconst(64, Type::I64),
            children: smallvec![],
        });
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![val, big],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(shl);
        // Shift by width: out of range, no known bits
        assert!(!kb.has_info());
    }

    #[test]
    fn propagate_shr_negative_amount_skipped() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(0xFF, Type::I64),
            children: smallvec![],
        });
        let neg = g.add(ENode {
            op: Op::Iconst(-1, Type::I64),
            children: smallvec![],
        });
        let shr = g.add(ENode {
            op: Op::Shr,
            children: smallvec![val, neg],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(shr);
        assert!(!kb.has_info());
    }

    // ── Sar propagation tests ───────────────────────────────────────────────

    #[test]
    fn propagate_sar_sign_bit_known_one() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        // 0xF000 in I16: sign bit (bit 15) is 1
        let val = g.add(ENode {
            op: Op::Iconst(0xF000u16 as i64, Type::I16),
            children: smallvec![],
        });
        let shift = g.add(ENode {
            op: Op::Iconst(4, Type::I16),
            children: smallvec![],
        });
        let sar = g.add(ENode {
            op: Op::Sar,
            children: smallvec![val, shift],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(sar);
        // Sar(0xF000, 4) in I16 = 0xFF00 (sign bit replicates into high 4 bits)
        let expected = 0xFF00u64 & 0xFFFF;
        assert_eq!(kb.known_ones & 0xFFFF, expected);
    }

    #[test]
    fn propagate_sar_sign_bit_known_zero() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        // 0x0F00 in I16: sign bit (bit 15) is 0
        let val = g.add(ENode {
            op: Op::Iconst(0x0F00, Type::I16),
            children: smallvec![],
        });
        let shift = g.add(ENode {
            op: Op::Iconst(4, Type::I16),
            children: smallvec![],
        });
        let sar = g.add(ENode {
            op: Op::Sar,
            children: smallvec![val, shift],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(sar);
        // Sar(0x0F00, 4) in I16 = 0x00F0 (zero-extend since sign bit is 0)
        assert_eq!(kb.known_ones & 0xFFFF, 0x00F0);
        // Upper 8 bits should be known-zero
        assert_eq!(kb.known_zeros & 0xFF00, 0xFF00);
    }

    #[test]
    fn propagate_sar_negative_amount_skipped() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(0xFF, Type::I64),
            children: smallvec![],
        });
        let neg = g.add(ENode {
            op: Op::Iconst(-1, Type::I64),
            children: smallvec![],
        });
        let sar = g.add(ENode {
            op: Op::Sar,
            children: smallvec![val, neg],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(sar);
        assert!(!kb.has_info());
    }

    // ── I32/I8 width tests ──────────────────────────────────────────────────

    #[test]
    fn propagate_and_i32() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let param = g.add(ENode {
            op: Op::Param(0, Type::I32),
            children: smallvec![],
        });
        let mask = g.add(ENode {
            op: Op::Iconst(0xF, Type::I32),
            children: smallvec![],
        });
        let and = g.add(ENode {
            op: Op::And,
            children: smallvec![param, mask],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(and);
        // Upper 28 bits should be known-zero (I32 mask 0xF)
        assert_eq!(kb.known_zeros & 0xFFFF_FFF0, 0xFFFF_FFF0);
    }

    #[test]
    fn propagate_shl_i8() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(1, Type::I8),
            children: smallvec![],
        });
        let shift = g.add(ENode {
            op: Op::Iconst(4, Type::I8),
            children: smallvec![],
        });
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![val, shift],
        });
        propagate_known_bits(&mut g);
        let kb = g.get_known_bits(shl);
        // 1 << 4 in I8 = 0x10
        assert_eq!(kb.known_ones, 0x10);
        // Lower 4 bits and bits 5-7 (except bit 4) should be known-zero
        assert_eq!(kb.known_zeros & 0xFF, 0xEF);
    }

    // ── Chained propagation tests ───────────────────────────────────────────

    #[test]
    fn propagate_chain_zext_then_and() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::egraph::phases::{CompileOptions, run_phases};
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let param = g.add(ENode {
            op: Op::Param(0, Type::I8),
            children: smallvec![],
        });
        let zext = g.add(ENode {
            op: Op::Zext(Type::I32),
            children: smallvec![param],
        });
        let mask = g.add(ENode {
            op: Op::Iconst(0xFFFF, Type::I32),
            children: smallvec![],
        });
        let and = g.add(ENode {
            op: Op::And,
            children: smallvec![zext, mask],
        });

        let opts = CompileOptions::default();
        run_phases(&mut g, &opts).expect("no blowup");

        // Zext(I32, param_i8) has bits 8-31 known-zero.
        // And(zext, 0xFFFF) clears bits 16-31 which are already zero.
        // So And is redundant.
        assert_eq!(g.find(and), g.find(zext));
    }

    // ── Known-constant promotion edge cases ─────────────────────────────────

    #[test]
    fn constant_promotion_i8() {
        use crate::egraph::egraph::EGraph;
        use crate::egraph::enode::ENode;
        use crate::egraph::phases::{CompileOptions, run_phases};
        use crate::ir::op::Op;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        // And(0xFF, 0x0F) in I8 = 0x0F
        let a = g.add(ENode {
            op: Op::Iconst(-1, Type::I8),
            children: smallvec![],
        });
        let b = g.add(ENode {
            op: Op::Iconst(0x0F, Type::I8),
            children: smallvec![],
        });
        let and = g.add(ENode {
            op: Op::And,
            children: smallvec![a, b],
        });

        let opts = CompileOptions::default();
        run_phases(&mut g, &opts).expect("no blowup");

        // Should fold to Iconst(0x0F, I8)
        let expected = g.add(ENode {
            op: Op::Iconst(0x0F, Type::I8),
            children: smallvec![],
        });
        assert_eq!(g.find(and), g.find(expected));
    }
}
