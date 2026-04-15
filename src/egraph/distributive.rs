use smallvec::smallvec;

use crate::egraph::egraph::{EGraph, NodeSnap, snapshot_all};
use crate::egraph::enode::ENode;
use crate::ir::op::{ClassId, Op};

/// Apply distributive factoring rules:
/// - Add(Mul(a,b), Mul(a,c)) = Mul(a, Add(b,c))  (shared factor)
/// - Sub(Mul(a,b), Mul(a,c)) = Mul(a, Sub(b,c))  (shared factor)
///
/// Only the factoring direction (reducing multiplications) is applied.
/// A blowup guard skips this pass if the e-graph is already 3/4 full.
pub fn apply_distributive_rules(egraph: &mut EGraph, max_classes: usize) -> bool {
    // Blowup guard: skip if e-graph is near capacity
    if egraph.class_count() > max_classes * 3 / 4 {
        return false;
    }

    let snaps = snapshot_all(egraph);
    let mut changed = false;
    changed |= apply_factoring(egraph, &snaps, true); // Add
    changed |= apply_factoring(egraph, &snaps, false); // Sub
    changed
}

/// Factor a shared Mul operand out of Add or Sub.
/// When `is_add` is true: Add(Mul(a,b), Mul(a,c)) = Mul(a, Add(b,c))
/// When `is_add` is false: Sub(Mul(a,b), Mul(a,c)) = Mul(a, Sub(b,c))
fn apply_factoring(egraph: &mut EGraph, snaps: &[NodeSnap], is_add: bool) -> bool {
    let mut changed = false;
    let target_op = if is_add { Op::Add } else { Op::Sub };

    for snap in snaps {
        if snap.op != target_op || snap.children.len() != 2 {
            continue;
        }
        let class_id = snap.class_id;
        let lhs_class_id = snap.children[0];
        let rhs_class_id = snap.children[1];

        let lhs_canon = egraph.unionfind.find_immutable(lhs_class_id);
        let rhs_canon = egraph.unionfind.find_immutable(rhs_class_id);

        // Collect Mul nodes from both children's classes
        let lhs_muls = collect_mul_pairs(egraph, lhs_canon);
        let rhs_muls = collect_mul_pairs(egraph, rhs_canon);

        if lhs_muls.is_empty() || rhs_muls.is_empty() {
            continue;
        }

        // Find a common factor between any pair
        let mut found = false;
        for &(la, lb) in &lhs_muls {
            for &(ra, rb) in &rhs_muls {
                // Check all 4 pairings of (la,lb) vs (ra,rb) for a common factor
                let common = find_common_factor(egraph, la, lb, ra, rb);
                if let Some((factor, other_lhs, other_rhs)) = common {
                    // Build: Mul(factor, Add/Sub(other_lhs, other_rhs))
                    let inner = egraph.add(ENode {
                        op: target_op.clone(),
                        children: smallvec![other_lhs, other_rhs],
                    });
                    let factored = egraph.add(ENode {
                        op: Op::Mul,
                        children: smallvec![factor, inner],
                    });
                    let factored_canon = egraph.unionfind.find_immutable(factored);
                    let class_canon = egraph.unionfind.find_immutable(class_id);
                    if factored_canon != class_canon {
                        egraph.merge(class_id, factored);
                        changed = true;
                    }
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
    }
    changed
}

/// Collect all (a, b) pairs from Mul nodes in a class.
fn collect_mul_pairs(egraph: &EGraph, class_id: ClassId) -> Vec<(ClassId, ClassId)> {
    let nodes = egraph.class(class_id).nodes.clone();
    let mut pairs = Vec::new();
    for node in &nodes {
        if node.op == Op::Mul && node.children.len() == 2 {
            let a = egraph.unionfind.find_immutable(node.children[0]);
            let b = egraph.unionfind.find_immutable(node.children[1]);
            pairs.push((a, b));
        }
    }
    pairs
}

/// Check if Mul(la, lb) and Mul(ra, rb) share a common canonical factor.
/// Returns Some((common_factor, other_from_lhs, other_from_rhs)) if found.
fn find_common_factor(
    egraph: &EGraph,
    la: ClassId,
    lb: ClassId,
    ra: ClassId,
    rb: ClassId,
) -> Option<(ClassId, ClassId, ClassId)> {
    let _ = egraph; // used for future extensibility; comparisons are on canonical ids
    // la == ra => factor=la, others=(lb, rb)
    if la == ra {
        return Some((la, lb, rb));
    }
    // la == rb => factor=la, others=(lb, ra)
    if la == rb {
        return Some((la, lb, ra));
    }
    // lb == ra => factor=lb, others=(la, rb)
    if lb == ra {
        return Some((lb, la, rb));
    }
    // lb == rb => factor=lb, others=(la, ra)
    if lb == rb {
        return Some((lb, la, ra));
    }
    None
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn iconst(g: &mut EGraph, v: i64) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(v, Type::I64),
            children: smallvec![],
        })
    }

    fn param(g: &mut EGraph, idx: u32) -> ClassId {
        g.add(ENode {
            op: Op::Param(idx, Type::I64),
            children: smallvec![],
        })
    }

    // Factor a on the left: Add(Mul(a, b), Mul(a, c)) => Mul(a, Add(b, c))
    #[test]
    fn factor_add_common_left() {
        let mut g = EGraph::new();
        let a = param(&mut g, 0);
        let b = param(&mut g, 1);
        let c = param(&mut g, 2);

        let mul_ab = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, b],
        });
        let mul_ac = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, c],
        });
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul_ab, mul_ac],
        });

        let changed = apply_distributive_rules(&mut g, 500_000);
        g.rebuild();

        assert!(changed, "factoring should have fired");

        // Verify Add(Mul(a,b), Mul(a,c)) equals Mul(a, Add(b,c))
        let inner = g.add(ENode {
            op: Op::Add,
            children: smallvec![b, c],
        });
        let factored = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, inner],
        });
        assert_eq!(
            g.find(add),
            g.find(factored),
            "Add(Mul(a,b), Mul(a,c)) should equal Mul(a, Add(b,c))"
        );
    }

    // Factor a on the right: Add(Mul(b, a), Mul(c, a)) => Mul(a, Add(b, c))
    #[test]
    fn factor_add_common_right() {
        let mut g = EGraph::new();
        let a = param(&mut g, 0);
        let b = param(&mut g, 1);
        let c = param(&mut g, 2);

        let mul_ba = g.add(ENode {
            op: Op::Mul,
            children: smallvec![b, a],
        });
        let mul_ca = g.add(ENode {
            op: Op::Mul,
            children: smallvec![c, a],
        });
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul_ba, mul_ca],
        });

        let changed = apply_distributive_rules(&mut g, 500_000);
        g.rebuild();

        assert!(changed, "factoring should have fired");

        // Verify equivalence: the Add class contains Mul(a, Add(b, c))
        let inner = g.add(ENode {
            op: Op::Add,
            children: smallvec![b, c],
        });
        let factored = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, inner],
        });
        assert_eq!(
            g.find(add),
            g.find(factored),
            "Add(Mul(b,a), Mul(c,a)) should equal Mul(a, Add(b,c))"
        );
    }

    // Sub factoring: Sub(Mul(a, b), Mul(a, c)) => Mul(a, Sub(b, c))
    #[test]
    fn factor_sub() {
        let mut g = EGraph::new();
        let a = param(&mut g, 0);
        let b = param(&mut g, 1);
        let c = param(&mut g, 2);

        let mul_ab = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, b],
        });
        let mul_ac = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, c],
        });
        let sub = g.add(ENode {
            op: Op::Sub,
            children: smallvec![mul_ab, mul_ac],
        });

        let changed = apply_distributive_rules(&mut g, 500_000);
        g.rebuild();

        assert!(changed, "factoring should have fired for Sub");

        let inner = g.add(ENode {
            op: Op::Sub,
            children: smallvec![b, c],
        });
        let factored = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, inner],
        });
        assert_eq!(
            g.find(sub),
            g.find(factored),
            "Sub(Mul(a,b), Mul(a,c)) should equal Mul(a, Sub(b,c))"
        );
    }

    // No match: all four operands distinct, no common factor
    #[test]
    fn no_match_different_factors() {
        let mut g = EGraph::new();
        let a = param(&mut g, 0);
        let b = param(&mut g, 1);
        let c = param(&mut g, 2);
        let d = param(&mut g, 3);

        let mul_ab = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, b],
        });
        let mul_cd = g.add(ENode {
            op: Op::Mul,
            children: smallvec![c, d],
        });
        let _add = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul_ab, mul_cd],
        });

        let changed = apply_distributive_rules(&mut g, 500_000);
        assert!(
            !changed,
            "no factoring should occur when all factors are distinct"
        );
    }

    // Cross-pairing: Add(Mul(a, b), Mul(c, a)) — factor on left-left and right-right
    #[test]
    fn factor_add_cross_pairing() {
        let mut g = EGraph::new();
        let a = param(&mut g, 0);
        let b = param(&mut g, 1);
        let c = param(&mut g, 2);

        let mul_ab = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, b],
        });
        let mul_ca = g.add(ENode {
            op: Op::Mul,
            children: smallvec![c, a],
        });
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul_ab, mul_ca],
        });

        let changed = apply_distributive_rules(&mut g, 500_000);
        g.rebuild();

        assert!(changed, "factoring should fire for cross-pairing");

        let inner = g.add(ENode {
            op: Op::Add,
            children: smallvec![b, c],
        });
        let factored = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, inner],
        });
        assert_eq!(
            g.find(add),
            g.find(factored),
            "Add(Mul(a,b), Mul(c,a)) should equal Mul(a, Add(b,c))"
        );
    }

    // Same factor: Add(Mul(a, a), Mul(a, b)) — a appears in both positions of left Mul
    #[test]
    fn factor_add_same_factor_squared() {
        let mut g = EGraph::new();
        let a = param(&mut g, 0);
        let b = param(&mut g, 1);

        let mul_aa = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, a],
        });
        let mul_ab = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, b],
        });
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul_aa, mul_ab],
        });

        let changed = apply_distributive_rules(&mut g, 500_000);
        g.rebuild();

        assert!(changed, "factoring should fire for squared factor");

        // a*a + a*b = a*(a+b)
        let inner = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, b],
        });
        let factored = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, inner],
        });
        assert_eq!(
            g.find(add),
            g.find(factored),
            "Add(Mul(a,a), Mul(a,b)) should equal Mul(a, Add(a,b))"
        );
    }

    // Blowup guard: returns false when e-graph exceeds 3/4 of max_classes
    #[test]
    fn blowup_guard_skips() {
        let mut g = EGraph::new();
        // Add a few nodes to make classes.len() > 0
        let _a = iconst(&mut g, 1);
        let _b = iconst(&mut g, 2);
        let _c = iconst(&mut g, 3);

        // max_classes=2 means threshold is 2*3/4 = 1; classes.len()=3 > 1, so skip
        let changed = apply_distributive_rules(&mut g, 2);
        assert!(
            !changed,
            "blowup guard should skip when e-graph is near capacity"
        );
    }
}
