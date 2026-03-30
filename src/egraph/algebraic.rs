use smallvec::smallvec;

use crate::egraph::egraph::{EGraph, snapshot_all};
use crate::egraph::enode::ENode;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// If `class_id` contains an Iconst node, return its value and type.
pub(crate) fn find_iconst(egraph: &EGraph, class_id: ClassId) -> Option<(i64, Type)> {
    if class_id == ClassId::NONE {
        return None;
    }
    let canon = egraph.unionfind.find_immutable(class_id);
    let class = egraph.class(canon);
    for node in &class.nodes {
        if let Op::Iconst(v, ref ty) = node.op {
            return Some((v, ty.clone()));
        }
    }
    None
}

fn make_iconst(egraph: &mut EGraph, val: i64, ty: Type) -> ClassId {
    egraph.add(ENode {
        op: Op::Iconst(val, ty),
        children: smallvec![],
    })
}

/// Mask a constant value to the type's bit width and sign-extend from the top bit,
/// so that i64 representation stays consistent with the narrower type.
/// For 64-bit types (or non-integer types), returns `val` unchanged.
fn mask_to_type(val: i64, ty: &Type) -> i64 {
    if !ty.is_integer() {
        return val;
    }
    let width = ty.bit_width();
    if width >= 64 {
        return val;
    }
    let mask = (1i64 << width) - 1;
    let masked = val & mask;
    // Sign-extend from the top bit of the narrower type.
    let sign_bit = 1i64 << (width - 1);
    if masked & sign_bit != 0 {
        masked | !mask
    } else {
        masked
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn apply_algebraic_rules(egraph: &mut EGraph) -> bool {
    let mut changed = false;
    changed |= apply_identity_rules(egraph);
    changed |= apply_annihilation_rules(egraph);
    changed |= apply_idempotence_rules(egraph);
    changed |= apply_inverse_rules(egraph);
    changed |= apply_double_negation_rules(egraph);
    changed |= apply_constant_folding(egraph);
    changed |= apply_commutativity_rules(egraph);
    changed
}

// ── Identity rules ────────────────────────────────────────────────────────────
// Add(a, 0) = a, Mul(a, 1) = a, Or(a, 0) = a, And(a, all_ones) = a

fn apply_identity_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        match &snap.op {
            Op::Add | Op::Or if snap.children.len() == 2 => {
                let [lhs, rhs] = [snap.children[0], snap.children[1]];
                // Add/Or(a, 0) = a
                if let Some((0, _)) = find_iconst(egraph, rhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let lhs_canon = egraph.unionfind.find_immutable(lhs);
                    if canon != lhs_canon {
                        egraph.merge(class_id, lhs);
                        changed = true;
                    }
                }
                if let Some((0, _)) = find_iconst(egraph, lhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let rhs_canon = egraph.unionfind.find_immutable(rhs);
                    if canon != rhs_canon {
                        egraph.merge(class_id, rhs);
                        changed = true;
                    }
                }
            }
            Op::Mul if snap.children.len() == 2 => {
                let [lhs, rhs] = [snap.children[0], snap.children[1]];
                // Mul(a, 1) = a
                if let Some((1, _)) = find_iconst(egraph, rhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let lhs_canon = egraph.unionfind.find_immutable(lhs);
                    if canon != lhs_canon {
                        egraph.merge(class_id, lhs);
                        changed = true;
                    }
                }
                if let Some((1, _)) = find_iconst(egraph, lhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let rhs_canon = egraph.unionfind.find_immutable(rhs);
                    if canon != rhs_canon {
                        egraph.merge(class_id, rhs);
                        changed = true;
                    }
                }
            }
            Op::And if snap.children.len() == 2 => {
                let [lhs, rhs] = [snap.children[0], snap.children[1]];
                // And(a, all_ones) = a — check for -1 (all ones in two's complement)
                if let Some((-1, _)) = find_iconst(egraph, rhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let lhs_canon = egraph.unionfind.find_immutable(lhs);
                    if canon != lhs_canon {
                        egraph.merge(class_id, lhs);
                        changed = true;
                    }
                }
                if let Some((-1, _)) = find_iconst(egraph, lhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let rhs_canon = egraph.unionfind.find_immutable(rhs);
                    if canon != rhs_canon {
                        egraph.merge(class_id, rhs);
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

// ── Annihilation rules ────────────────────────────────────────────────────────
// Mul(a, 0) = 0, And(a, 0) = 0

fn apply_annihilation_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        match &snap.op {
            Op::Mul | Op::And if snap.children.len() == 2 => {
                let [lhs, rhs] = [snap.children[0], snap.children[1]];
                let zero_side = if let Some((0, ref ty)) = find_iconst(egraph, rhs) {
                    Some((0i64, ty.clone()))
                } else if let Some((0, ref ty)) = find_iconst(egraph, lhs) {
                    Some((0i64, ty.clone()))
                } else {
                    None
                };
                if let Some((_, ty)) = zero_side {
                    let zero_class = make_iconst(egraph, 0, ty);
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let zero_canon = egraph.unionfind.find_immutable(zero_class);
                    if canon != zero_canon {
                        egraph.merge(class_id, zero_class);
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

// ── Idempotence rules ─────────────────────────────────────────────────────────
// And(a, a) = a, Or(a, a) = a

fn apply_idempotence_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        match &snap.op {
            Op::And | Op::Or if snap.children.len() == 2 => {
                let lhs_canon = egraph.unionfind.find_immutable(snap.children[0]);
                let rhs_canon = egraph.unionfind.find_immutable(snap.children[1]);
                if lhs_canon == rhs_canon {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if canon != lhs_canon {
                        egraph.merge(class_id, lhs_canon);
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

// ── Inverse rules ─────────────────────────────────────────────────────────────
// Sub(a, a) = 0, Xor(a, a) = 0

fn apply_inverse_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        match &snap.op {
            Op::Sub | Op::Xor if snap.children.len() == 2 => {
                let lhs_canon = egraph.unionfind.find_immutable(snap.children[0]);
                let rhs_canon = egraph.unionfind.find_immutable(snap.children[1]);
                if lhs_canon == rhs_canon {
                    // Derive type from the class's type
                    let ty = egraph
                        .class(egraph.unionfind.find_immutable(class_id))
                        .ty
                        .clone();
                    let zero_class = make_iconst(egraph, 0, ty);
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let zero_canon = egraph.unionfind.find_immutable(zero_class);
                    if canon != zero_canon {
                        egraph.merge(class_id, zero_class);
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

// ── Double negation ───────────────────────────────────────────────────────────
// Sub(0, Sub(0, a)) = a

fn apply_double_negation_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        // Outer Sub(0, ?)
        if snap.op != Op::Sub || snap.children.len() != 2 {
            continue;
        }
        if find_iconst(egraph, snap.children[0]).map(|(v, _)| v) != Some(0) {
            continue;
        }
        // Inner child must be Sub(0, a)
        let inner_canon = egraph.unionfind.find_immutable(snap.children[1]);
        let inner_class = egraph.class(inner_canon);
        for inner_node in inner_class.nodes.clone() {
            if inner_node.op != Op::Sub || inner_node.children.len() != 2 {
                continue;
            }
            if find_iconst(egraph, inner_node.children[0]).map(|(v, _)| v) != Some(0) {
                continue;
            }
            // Found Sub(0, Sub(0, a)); merge outer with a
            let a = inner_node.children[1];
            let a_canon = egraph.unionfind.find_immutable(a);
            let outer_canon = egraph.unionfind.find_immutable(class_id);
            if outer_canon != a_canon {
                egraph.merge(class_id, a);
                changed = true;
            }
        }
    }
    changed
}

// ── Constant folding ──────────────────────────────────────────────────────────

pub fn apply_constant_folding(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let lhs = snap.children[0];
        let rhs = snap.children[1];
        let Some((lv, lty)) = find_iconst(egraph, lhs) else {
            continue;
        };
        let Some((rv, _rty)) = find_iconst(egraph, rhs) else {
            continue;
        };

        let result: Option<i64> = match &snap.op {
            Op::Add => Some(lv.wrapping_add(rv)),
            Op::Sub => Some(lv.wrapping_sub(rv)),
            Op::Mul => Some(lv.wrapping_mul(rv)),
            Op::UDiv if rv != 0 => Some(((lv as u64).wrapping_div(rv as u64)) as i64),
            Op::SDiv if rv != 0 => Some(lv.wrapping_div(rv)),
            Op::URem if rv != 0 => Some(((lv as u64).wrapping_rem(rv as u64)) as i64),
            Op::SRem if rv != 0 => Some(lv.wrapping_rem(rv)),
            Op::And => Some(lv & rv),
            Op::Or => Some(lv | rv),
            Op::Xor => Some(lv ^ rv),
            Op::Shl => Some(lv.wrapping_shl(rv as u32)),
            Op::Shr => Some(((lv as u64).wrapping_shr(rv as u32)) as i64),
            Op::Sar => Some(lv.wrapping_shr(rv as u32)),
            _ => None,
        };

        if let Some(folded) = result {
            let folded = mask_to_type(folded, &lty);
            let result_class = make_iconst(egraph, folded, lty);
            let canon = egraph.unionfind.find_immutable(class_id);
            let result_canon = egraph.unionfind.find_immutable(result_class);
            if canon != result_canon {
                egraph.merge(class_id, result_class);
                changed = true;
            }
        }
    }
    changed
}

// ── Commutativity rules ───────────────────────────────────────────────────────
// Add(a,b) = Add(b,a), Mul(a,b) = Mul(b,a), And/Or/Xor similarly.
// Only add swapped version if child[0].0 > child[1].0 to avoid infinite loops.

fn apply_commutativity_rules(egraph: &mut EGraph) -> bool {
    // Spike test note: this uses canonical-id ordering to ensure each pair is
    // added at most once, preventing combinatorial blowup.  A chain of 10+ Add
    // nodes with commutativity enabled stays well under 50k classes because each
    // Add(a,b) class gains at most one new node (the swapped one), and only when
    // the left child's canonical id is strictly greater than the right child's.
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        if snap.children.len() != 2 {
            continue;
        }
        let commutative = matches!(&snap.op, Op::Add | Op::Mul | Op::And | Op::Or | Op::Xor);
        if !commutative {
            continue;
        }

        let a = egraph.unionfind.find_immutable(snap.children[0]);
        let b = egraph.unionfind.find_immutable(snap.children[1]);

        // Only add the commuted version when a > b (canonical ordering guard)
        if a.0 > b.0 {
            let swapped = egraph.add(ENode {
                op: snap.op.clone(),
                children: smallvec![b, a],
            });
            let class_canon = egraph.unionfind.find_immutable(snap.class_id);
            let swapped_canon = egraph.unionfind.find_immutable(swapped);
            if class_canon != swapped_canon {
                egraph.merge(snap.class_id, swapped);
                changed = true;
            }
        }
    }
    changed
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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

    fn add(g: &mut EGraph, a: ClassId, b: ClassId) -> ClassId {
        g.add(ENode {
            op: Op::Add,
            children: smallvec![a, b],
        })
    }

    fn mul(g: &mut EGraph, a: ClassId, b: ClassId) -> ClassId {
        g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, b],
        })
    }

    fn xor(g: &mut EGraph, a: ClassId, b: ClassId) -> ClassId {
        g.add(ENode {
            op: Op::Xor,
            children: smallvec![a, b],
        })
    }

    // 4.9: Add(x, 0) merges with x
    #[test]
    fn add_identity_zero() {
        let mut g = EGraph::new();
        let x = iconst(&mut g, 5, Type::I64);
        let zero = iconst(&mut g, 0, Type::I64);
        let result = add(&mut g, x, zero);
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(result), g.find(x));
    }

    // 4.9: Mul(x, 0) merges with Iconst(0)
    #[test]
    fn mul_annihilation_zero() {
        let mut g = EGraph::new();
        let x = iconst(&mut g, 5, Type::I64);
        let zero = iconst(&mut g, 0, Type::I64);
        let result = mul(&mut g, x, zero);
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(result), g.find(zero));
    }

    // 4.9: Xor(x, x) merges with Iconst(0)
    #[test]
    fn xor_inverse() {
        let mut g = EGraph::new();
        let x = iconst(&mut g, 42, Type::I64);
        let result = xor(&mut g, x, x);
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let zero = iconst(&mut g, 0, Type::I64);
        assert_eq!(g.find(result), g.find(zero));
    }

    // 4.9: constant fold Add(Iconst(3), Iconst(7)) => Iconst(10)
    #[test]
    fn constant_fold_add() {
        let mut g = EGraph::new();
        let c3 = iconst(&mut g, 3, Type::I64);
        let c7 = iconst(&mut g, 7, Type::I64);
        let result = add(&mut g, c3, c7);
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let ten = iconst(&mut g, 10, Type::I64);
        assert_eq!(g.find(result), g.find(ten));
    }

    // 4.7a: commutativity spike — Add chain of depth 10 stays reasonable
    // With canonical-ordering guard each Add(a,b) class gets at most 1 extra node,
    // so total classes <= 2*nodes, never approaches 50k for small programs.
    #[test]
    fn commutativity_spike_depth10() {
        let mut g = EGraph::new();
        let base = g.add(ENode {
            op: Op::Iconst(1, Type::I64),
            children: smallvec![],
        });
        let mut cur = base;
        for i in 2..=11i64 {
            let c = g.add(ENode {
                op: Op::Iconst(i, Type::I64),
                children: smallvec![],
            });
            cur = g.add(ENode {
                op: Op::Add,
                children: smallvec![cur, c],
            });
        }
        apply_commutativity_rules(&mut g);
        g.rebuild();
        let count = g.class_count();
        assert!(
            count < 50_000,
            "e-class count {count} exceeded blowup threshold"
        );
    }
}
