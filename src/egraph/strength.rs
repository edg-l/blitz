use smallvec::smallvec;

use crate::egraph::egraph::EGraph;
use crate::egraph::enode::ENode;
use crate::ir::op::{ClassId, Op};
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
            Op::Mul if children.len() == 2 => {
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
                        op: Op::Iconst(n, ty.clone()),
                        children: smallvec![],
                    });
                    let shl = egraph.add(ENode {
                        op: Op::Shl,
                        children: smallvec![non_const, n_class],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(shl) != canon {
                        egraph.merge(class_id, shl);
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
                        op: Op::Iconst(n, ty),
                        children: smallvec![],
                    });
                    let shl = egraph.add(ENode {
                        op: Op::Shl,
                        children: smallvec![non_const, n_class],
                    });
                    let sum = egraph.add(ENode {
                        op: Op::Add,
                        children: smallvec![non_const, shl],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(sum) != canon {
                        egraph.merge(class_id, sum);
                        changed = true;
                    }
                }
            }

            Op::UDiv if children.len() == 2 => {
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
                        op: Op::Iconst(n, ty),
                        children: smallvec![],
                    });
                    let shr = egraph.add(ENode {
                        op: Op::Shr,
                        children: smallvec![a, n_class],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(shr) != canon {
                        egraph.merge(class_id, shr);
                        changed = true;
                    }
                }
            }

            Op::URem if children.len() == 2 => {
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
                        op: Op::Iconst(mask, ty),
                        children: smallvec![],
                    });
                    let and = egraph.add(ENode {
                        op: Op::And,
                        children: smallvec![a, mask_class],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(and) != canon {
                        egraph.merge(class_id, and);
                        changed = true;
                    }
                }
            }

            Op::SDiv if children.len() == 2 => {
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
                        op: Op::Iconst(63, Type::I64),
                        children: smallvec![],
                    });
                    let c64_minus_n = egraph.add(ENode {
                        op: Op::Iconst(64 - n, Type::I64),
                        children: smallvec![],
                    });
                    let cn = egraph.add(ENode {
                        op: Op::Iconst(n, Type::I64),
                        children: smallvec![],
                    });
                    // Sar(a, 63)
                    let sar63 = egraph.add(ENode {
                        op: Op::Sar,
                        children: smallvec![a, c63],
                    });
                    // Shr(Sar(a, 63), 64-n)
                    let shr_adj = egraph.add(ENode {
                        op: Op::Shr,
                        children: smallvec![sar63, c64_minus_n],
                    });
                    // Add(a, Shr(...))
                    let adj_add = egraph.add(ENode {
                        op: Op::Add,
                        children: smallvec![a, shr_adj],
                    });
                    // Sar(Add(...), n)
                    let result = egraph.add(ENode {
                        op: Op::Sar,
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

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;

    fn iconst(g: &mut EGraph, v: i64, ty: Type) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(v, ty),
            children: smallvec![],
        })
    }

    // 4.11: Mul(a, 8) => Shl(a, 3)
    #[test]
    fn mul_pow2_becomes_shl() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Iconst(99, Type::I64),
            children: smallvec![],
        });
        let eight = iconst(&mut g, 8, Type::I64);
        let mul = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, eight],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        // Shl(a, 3) should be in same class as mul
        let three = iconst(&mut g, 3, Type::I64);
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, three],
        });
        assert_eq!(g.find(mul), g.find(shl));
    }

    // 4.11: UDiv(a, 4) => Shr(a, 2)
    #[test]
    fn udiv_pow2_becomes_shr() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Iconst(100, Type::I64),
            children: smallvec![],
        });
        let four = iconst(&mut g, 4, Type::I64);
        let udiv = g.add(ENode {
            op: Op::UDiv,
            children: smallvec![a, four],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        let two = iconst(&mut g, 2, Type::I64);
        let shr = g.add(ENode {
            op: Op::Shr,
            children: smallvec![a, two],
        });
        assert_eq!(g.find(udiv), g.find(shr));
    }

    // 4.11: URem(a, 8) => And(a, 7)
    #[test]
    fn urem_pow2_becomes_and() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Iconst(200, Type::I64),
            children: smallvec![],
        });
        let eight = iconst(&mut g, 8, Type::I64);
        let urem = g.add(ENode {
            op: Op::URem,
            children: smallvec![a, eight],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        let seven = iconst(&mut g, 7, Type::I64);
        let and = g.add(ENode {
            op: Op::And,
            children: smallvec![a, seven],
        });
        assert_eq!(g.find(urem), g.find(and));
    }

    // 4.11: Mul(a, 3) => Add(a, Shl(a, 1))
    #[test]
    fn mul_by_3_lea_form() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Iconst(7, Type::I64),
            children: smallvec![],
        });
        let three = iconst(&mut g, 3, Type::I64);
        let mul = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, three],
        });
        apply_strength_reduction(&mut g);
        g.rebuild();

        let one = iconst(&mut g, 1, Type::I64);
        let shl1 = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, one],
        });
        let sum = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, shl1],
        });
        assert_eq!(g.find(mul), g.find(sum));
    }

    // 4.11: SDiv(a, 4, I64) produces the signed div pattern
    #[test]
    fn sdiv_pow2_signed_pattern() {
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let four = iconst(&mut g, 4, Type::I64);
        let sdiv = g.add(ENode {
            op: Op::SDiv,
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
