use smallvec::smallvec;

use crate::egraph::egraph::EGraph;
use crate::egraph::enode::ENode;
use crate::ir::op::{ClassId, Op, PureOp};
use crate::ir::types::Type;

pub fn apply_strength_reduction(egraph: &mut EGraph) -> bool {
    let mut changed = false;

    // Snapshot class ids and their nodes first.
    let snaps: Vec<(ClassId, Op, smallvec::SmallVec<[ClassId; 2]>)> = {
        let mut v = Vec::new();
        for i in 0..egraph.classes.len() as u32 {
            let id = ClassId(i);
            if egraph.unionfind.find_immutable(id) != id {
                continue;
            }
            let class = egraph.class(id);
            for node in &class.nodes {
                v.push((id, node.op.clone(), node.children.clone()));
            }
        }
        v
    };

    for (class_id, op, children) in &snaps {
        let class_id = *class_id;
        match op {
            Op::Pure(PureOp::Mul) if children.len() == 2 => {
                let a = children[0];
                let b = children[1];
                // Check both orderings for the constant
                let (val_opt, non_const) = if let Some((v, _ty)) = egraph.get_constant(b) {
                    (Some(v), a)
                } else if let Some((v, _ty)) = egraph.get_constant(a) {
                    (Some(v), b)
                } else {
                    (None, a)
                };

                let Some(val) = val_opt else { continue };

                // Mul(a, 2^n) = Shl(a, n)
                if val > 0 && val.count_ones() == 1 {
                    let n = val.trailing_zeros() as i64;
                    let ty = egraph
                        .class(egraph.unionfind.find_immutable(class_id))
                        .ty
                        .clone();
                    let n_class = egraph.add(ENode {
                        op: Op::Pure(PureOp::Iconst(n, ty.clone())),
                        children: smallvec![],
                    });
                    let shl = egraph.add(ENode {
                        op: Op::Pure(PureOp::Shl),
                        children: smallvec![non_const, n_class],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(shl) != canon {
                        egraph.merge(class_id, shl);
                        changed = true;
                    }
                }

                if let Some(alt) = decompose_const_mul(egraph, class_id, non_const, val) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(alt) != canon {
                        egraph.merge(class_id, alt);
                        changed = true;
                    }
                }

                // Mul(a, 3/5/9) = Add(a, Shl(a, 1/2/3))
                let shift_for_mul: Option<i64> = match val {
                    3 => Some(1),
                    5 => Some(2),
                    9 => Some(3),
                    _ => None,
                };
                if let Some(n) = shift_for_mul {
                    let ty = egraph
                        .class(egraph.unionfind.find_immutable(class_id))
                        .ty
                        .clone();
                    let n_class = egraph.add(ENode {
                        op: Op::Pure(PureOp::Iconst(n, ty)),
                        children: smallvec![],
                    });
                    let shl = egraph.add(ENode {
                        op: Op::Pure(PureOp::Shl),
                        children: smallvec![non_const, n_class],
                    });
                    let sum = egraph.add(ENode {
                        op: Op::Pure(PureOp::Add),
                        children: smallvec![non_const, shl],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(sum) != canon {
                        egraph.merge(class_id, sum);
                        changed = true;
                    }
                }
            }

            Op::Pure(PureOp::UDiv) if children.len() == 2 => {
                let a = children[0];
                let b = children[1];
                let Some((val, _ty)) = egraph.get_constant(b) else {
                    continue;
                };
                // UDiv(a, 2^n) = Shr(a, n)
                if val > 0 && val.count_ones() == 1 {
                    let n = val.trailing_zeros() as i64;
                    let ty = egraph
                        .class(egraph.unionfind.find_immutable(class_id))
                        .ty
                        .clone();
                    let n_class = egraph.add(ENode {
                        op: Op::Pure(PureOp::Iconst(n, ty)),
                        children: smallvec![],
                    });
                    let shr = egraph.add(ENode {
                        op: Op::Pure(PureOp::Shr),
                        children: smallvec![a, n_class],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(shr) != canon {
                        egraph.merge(class_id, shr);
                        changed = true;
                    }
                }
            }

            Op::Pure(PureOp::URem) if children.len() == 2 => {
                let a = children[0];
                let b = children[1];
                let Some((val, _ty)) = egraph.get_constant(b) else {
                    continue;
                };
                // URem(a, 2^n) = And(a, 2^n - 1)
                if val > 0 && val.count_ones() == 1 {
                    let mask = val.wrapping_sub(1);
                    let ty = egraph
                        .class(egraph.unionfind.find_immutable(class_id))
                        .ty
                        .clone();
                    let mask_class = egraph.add(ENode {
                        op: Op::Pure(PureOp::Iconst(mask, ty)),
                        children: smallvec![],
                    });
                    let and = egraph.add(ENode {
                        op: Op::Pure(PureOp::And),
                        children: smallvec![a, mask_class],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(and) != canon {
                        egraph.merge(class_id, and);
                        changed = true;
                    }
                }
            }

            Op::Pure(PureOp::SDiv) if children.len() == 2 => {
                let a = children[0];
                let b = children[1];
                let Some((val, ty)) = egraph.get_constant(b) else {
                    continue;
                };
                // SDiv(a, 2^n) for I64: Sar(Add(a, Shr(Sar(a, 63), 64-n)), n)
                // val == 1 would produce Shr(_, 64) which is UB on x86-64 (shift masked to 63).
                if ty == Type::I64 && val > 1 && val.count_ones() == 1 {
                    let n = val.trailing_zeros() as i64;
                    let c63 = egraph.add(ENode {
                        op: Op::Pure(PureOp::Iconst(63, Type::I64)),
                        children: smallvec![],
                    });
                    let c64_minus_n = egraph.add(ENode {
                        op: Op::Pure(PureOp::Iconst(64 - n, Type::I64)),
                        children: smallvec![],
                    });
                    let cn = egraph.add(ENode {
                        op: Op::Pure(PureOp::Iconst(n, Type::I64)),
                        children: smallvec![],
                    });
                    // Sar(a, 63)
                    let sar63 = egraph.add(ENode {
                        op: Op::Pure(PureOp::Sar),
                        children: smallvec![a, c63],
                    });
                    // Shr(Sar(a, 63), 64-n)
                    let shr_adj = egraph.add(ENode {
                        op: Op::Pure(PureOp::Shr),
                        children: smallvec![sar63, c64_minus_n],
                    });
                    // Add(a, Shr(...))
                    let adj_add = egraph.add(ENode {
                        op: Op::Pure(PureOp::Add),
                        children: smallvec![a, shr_adj],
                    });
                    // Sar(Add(...), n)
                    let result = egraph.add(ENode {
                        op: Op::Pure(PureOp::Sar),
                        children: smallvec![adj_add, cn],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(result) != canon {
                        egraph.merge(class_id, result);
                        changed = true;
                    }
                }
            }

            _ => {}
        }
    }
    changed
}

/// A shift-and-add form of `x * k`, or `None` where there is no short one.
///
/// This is the one rule in the set whose result is not unambiguously better:
/// `imul` is a single instruction, and the decomposition is two, so which wins
/// is a question about their relative cost rather than about the shape of the
/// expression. It is added as an alternative and extraction decides, which is
/// what an e-graph is for. Every other rule here would be sound to apply
/// destructively.
///
/// Three forms, all of them two instructions:
///
/// - `k = m << n` for `m` in 3, 5, 9 -- `(x * m) << n`, where the inner
///   multiply is the `lea` form the rule below produces. `x * 12` is
///   `lea (x + x*2)` then `shl 2`, which is what gcc emits.
/// - `k = 2^n - 1` -- `(x << n) - x`.
/// - `k = 2^n + 1` -- `(x << n) + x`.
///
/// Deliberately not covered: an arbitrary two-bit `k` as `(x << a) + (x << b)`.
/// That is three instructions against `imul`'s one, and at this cost model it
/// loses, so adding it would widen every such class to no purpose. The bound on
/// this rule is that it fires only on a constant multiplier and emits a fixed
/// shape, so no chain of it can compound -- unlike reassociation, whose
/// associations of an n-term chain are Catalan-many.
fn decompose_const_mul(
    egraph: &mut EGraph,
    class_id: ClassId,
    x: ClassId,
    k: i64,
) -> Option<ClassId> {
    let ty = egraph
        .class(egraph.unionfind.find_immutable(class_id))
        .ty
        .clone();
    if !ty.is_integer() {
        return None;
    }
    let width = i64::from(ty.bit_width());
    if k <= 0 {
        return None;
    }

    // A shift by the type's own width or more is undefined, and a decomposition
    // is only worth having while it stays shorter than the multiply.
    let shift_ok = |n: i64| n > 0 && n < width;

    let shl = |egraph: &mut EGraph, val: ClassId, n: i64| {
        let n_class = egraph.add(ENode {
            op: Op::Pure(PureOp::Iconst(n, ty.clone())),
            children: smallvec![],
        });
        egraph.add(ENode {
            op: Op::Pure(PureOp::Shl),
            children: smallvec![val, n_class],
        })
    };

    // k = m << n, m in {3,5,9}: the inner multiply becomes a `lea` on its own.
    let n = k.trailing_zeros() as i64;
    let odd = k >> n;
    if shift_ok(n) && matches!(odd, 3 | 5 | 9) {
        let m = egraph.add(ENode {
            op: Op::Pure(PureOp::Iconst(odd, ty.clone())),
            children: smallvec![],
        });
        let inner = egraph.add(ENode {
            op: Op::Pure(PureOp::Mul),
            children: smallvec![x, m],
        });
        return Some(shl(egraph, inner, n));
    }

    // k = 2^n - 1, and k = 2^n + 1 beyond the 3/5/9 the `lea` rule already has.
    for (delta, op) in [(1, PureOp::Sub), (-1, PureOp::Add)] {
        let pow = k + delta;
        if pow > 0 && pow.count_ones() == 1 {
            let n = pow.trailing_zeros() as i64;
            if shift_ok(n) {
                let shifted = shl(egraph, x, n);
                return Some(egraph.add(ENode {
                    op: Op::Pure(op),
                    children: smallvec![shifted, x],
                }));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;

    fn iconst(g: &mut EGraph, v: i64, ty: Type) -> ClassId {
        g.add(ENode {
            op: Op::Pure(PureOp::Iconst(v, ty)),
            children: smallvec![],
        })
    }

    // 4.11: Mul(a, 8) => Shl(a, 3)
    #[test]
    fn mul_pow2_becomes_shl() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(99, Type::I64)),
            children: smallvec![],
        });
        let eight = iconst(&mut g, 8, Type::I64);
        let mul = g.add(ENode {
            op: Op::Pure(PureOp::Mul),
            children: smallvec![a, eight],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        // Shl(a, 3) should be in same class as mul
        let three = iconst(&mut g, 3, Type::I64);
        let shl = g.add(ENode {
            op: Op::Pure(PureOp::Shl),
            children: smallvec![a, three],
        });
        assert_eq!(g.find(mul), g.find(shl));
    }

    // 4.11: UDiv(a, 4) => Shr(a, 2)
    #[test]
    fn udiv_pow2_becomes_shr() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(100, Type::I64)),
            children: smallvec![],
        });
        let four = iconst(&mut g, 4, Type::I64);
        let udiv = g.add(ENode {
            op: Op::Pure(PureOp::UDiv),
            children: smallvec![a, four],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        let two = iconst(&mut g, 2, Type::I64);
        let shr = g.add(ENode {
            op: Op::Pure(PureOp::Shr),
            children: smallvec![a, two],
        });
        assert_eq!(g.find(udiv), g.find(shr));
    }

    // 4.11: URem(a, 8) => And(a, 7)
    #[test]
    fn urem_pow2_becomes_and() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(200, Type::I64)),
            children: smallvec![],
        });
        let eight = iconst(&mut g, 8, Type::I64);
        let urem = g.add(ENode {
            op: Op::Pure(PureOp::URem),
            children: smallvec![a, eight],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        let seven = iconst(&mut g, 7, Type::I64);
        let and = g.add(ENode {
            op: Op::Pure(PureOp::And),
            children: smallvec![a, seven],
        });
        assert_eq!(g.find(urem), g.find(and));
    }

    // 4.11: Mul(a, 3) => Add(a, Shl(a, 1))
    #[test]
    fn mul_by_3_lea_form() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(7, Type::I64)),
            children: smallvec![],
        });
        let three = iconst(&mut g, 3, Type::I64);
        let mul = g.add(ENode {
            op: Op::Pure(PureOp::Mul),
            children: smallvec![a, three],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        let one = iconst(&mut g, 1, Type::I64);
        let shl1 = g.add(ENode {
            op: Op::Pure(PureOp::Shl),
            children: smallvec![a, one],
        });
        let sum = g.add(ENode {
            op: Op::Pure(PureOp::Add),
            children: smallvec![a, shl1],
        });
        assert_eq!(g.find(mul), g.find(sum));
    }

    // 4.11: SDiv(a, 4, I64) produces the signed div pattern
    #[test]
    fn sdiv_pow2_signed_pattern() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(1, Type::I64)),
            children: smallvec![],
        });
        let four = iconst(&mut g, 4, Type::I64);
        let sdiv = g.add(ENode {
            op: Op::Pure(PureOp::SDiv),
            children: smallvec![a, four],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        // Just verify the sdiv class gained nodes (it merged with the pattern)
        let canon = g.find(sdiv);
        let class = g.class(canon);
        assert!(
            class.nodes.len() > 1,
            "SDiv class should have the pattern as an equivalent"
        );
    }
}
