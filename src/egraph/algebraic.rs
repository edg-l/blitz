use std::collections::BTreeMap;

use smallvec::smallvec;

use crate::egraph::egraph::{EGraph, NodeSnap, snapshot_all};
use crate::egraph::enode::ENode;
use crate::ir::condcode::CondCode;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

// ── Helpers ───────────────────────────────────────────────────────────────────

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
    let snaps = snapshot_all(egraph);
    let mut changed = false;
    changed |= apply_identity_rules(egraph, &snaps);
    changed |= apply_annihilation_rules(egraph, &snaps);
    changed |= apply_idempotence_rules(egraph, &snaps);
    changed |= apply_inverse_rules(egraph, &snaps);
    changed |= apply_double_negation_rules(egraph, &snaps);
    changed |= apply_constant_folding(egraph, &snaps);
    changed |= apply_commutativity_rules(egraph, &snaps);
    changed |= apply_reassociation_rules(egraph, &snaps);
    changed |= apply_shift_combining_rules(egraph, &snaps);
    changed |= apply_absorption_rules(egraph, &snaps);
    changed |= apply_div_identity_rules(egraph, &snaps);
    changed |= apply_select_rules(egraph, &snaps);
    changed |= apply_comparison_select_folding(egraph, &snaps);
    changed |= apply_extension_folding_rules(egraph, &snaps);
    changed |= apply_complement_rules(egraph, &snaps);
    changed |= apply_demorgan_rules(egraph, &snaps);
    changed |= apply_negation_distribution_rules(egraph, &snaps);
    changed |= apply_sub_zero_eq_ne_rules(egraph, &snaps);
    changed
}

/// `Icmp(Eq/Ne, Sub(a, b), 0)` → `Icmp(Eq/Ne, a, b)` (and the symmetric form).
///
/// Only `Eq`/`Ne` — the rewrite is unsafe for signed ordering comparisons
/// because `a - b` can overflow, so `Sgt((a - b), 0)` differs from `Sgt(a, b)`
/// when overflow occurs. Equality is immune: `(a - b) == 0` iff `a == b` in
/// wrapping arithmetic.
fn apply_sub_zero_eq_ne_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    use crate::ir::condcode::CondCode;

    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        let Op::Icmp(cc) = &snap.op else { continue };
        if !matches!(cc, CondCode::Eq | CondCode::Ne) {
            continue;
        }
        if snap.children.len() != 2 {
            continue;
        }
        let lhs = snap.children[0];
        let rhs = snap.children[1];

        // Try (Sub(a, b), 0); also accept (0, Sub(a, b)) because Eq/Ne are
        // symmetric in operand order.
        let (sub_side, other) = if egraph.get_constant(rhs).map(|(v, _)| v) == Some(0) {
            (lhs, rhs)
        } else if egraph.get_constant(lhs).map(|(v, _)| v) == Some(0) {
            (rhs, lhs)
        } else {
            continue;
        };
        let _ = other; // retained for clarity; we only needed to confirm the zero side

        let sub_canon = egraph.unionfind.find_immutable(sub_side);
        let sub_nodes = egraph.class(sub_canon).nodes.clone();
        for node in sub_nodes {
            if node.op != Op::Sub || node.children.len() != 2 {
                continue;
            }
            let a = node.children[0];
            let b = node.children[1];
            let new_icmp = egraph.add(ENode {
                op: Op::Icmp(*cc),
                children: smallvec![a, b],
            });
            let canon = egraph.unionfind.find_immutable(class_id);
            let new_canon = egraph.unionfind.find_immutable(new_icmp);
            if canon != new_canon {
                egraph.merge(class_id, new_icmp);
                changed = true;
            }
            break;
        }
    }
    changed
}

// ── Identity rules ────────────────────────────────────────────────────────────
// Add(a, 0) = a, Mul(a, 1) = a, Or(a, 0) = a, And(a, all_ones) = a

fn apply_identity_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        match &snap.op {
            Op::Add | Op::Or if snap.children.len() == 2 => {
                let [lhs, rhs] = [snap.children[0], snap.children[1]];
                // Add/Or(a, 0) = a
                if let Some((0, _)) = egraph.get_constant(rhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let lhs_canon = egraph.unionfind.find_immutable(lhs);
                    if canon != lhs_canon {
                        egraph.merge(class_id, lhs);
                        changed = true;
                    }
                }
                if let Some((0, _)) = egraph.get_constant(lhs) {
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
                if let Some((1, _)) = egraph.get_constant(rhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let lhs_canon = egraph.unionfind.find_immutable(lhs);
                    if canon != lhs_canon {
                        egraph.merge(class_id, lhs);
                        changed = true;
                    }
                }
                if let Some((1, _)) = egraph.get_constant(lhs) {
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
                if let Some((-1, _)) = egraph.get_constant(rhs) {
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let lhs_canon = egraph.unionfind.find_immutable(lhs);
                    if canon != lhs_canon {
                        egraph.merge(class_id, lhs);
                        changed = true;
                    }
                }
                if let Some((-1, _)) = egraph.get_constant(lhs) {
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
// Mul(a, 0) = 0, And(a, 0) = 0, Or(a, -1) = -1

fn apply_annihilation_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        match &snap.op {
            Op::Mul | Op::And if snap.children.len() == 2 => {
                let [lhs, rhs] = [snap.children[0], snap.children[1]];
                let zero_side = if let Some((0, ref ty)) = egraph.get_constant(rhs) {
                    Some((0i64, ty.clone()))
                } else if let Some((0, ref ty)) = egraph.get_constant(lhs) {
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
            // Or(a, -1) = -1
            Op::Or if snap.children.len() == 2 => {
                let [lhs, rhs] = [snap.children[0], snap.children[1]];
                let all_ones = if let Some((-1, ref ty)) = egraph.get_constant(rhs) {
                    Some(ty.clone())
                } else if let Some((-1, ref ty)) = egraph.get_constant(lhs) {
                    Some(ty.clone())
                } else {
                    None
                };
                if let Some(ty) = all_ones {
                    let ones_class = make_iconst(egraph, -1, ty);
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let ones_canon = egraph.unionfind.find_immutable(ones_class);
                    if canon != ones_canon {
                        egraph.merge(class_id, ones_class);
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

fn apply_idempotence_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
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

fn apply_inverse_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
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

fn apply_double_negation_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        // Outer Sub(0, ?)
        if snap.op != Op::Sub || snap.children.len() != 2 {
            continue;
        }
        if egraph.get_constant(snap.children[0]).map(|(v, _)| v) != Some(0) {
            continue;
        }
        // Inner child must be Sub(0, a)
        let inner_canon = egraph.unionfind.find_immutable(snap.children[1]);
        let inner_class = egraph.class(inner_canon);
        for inner_node in inner_class.nodes.clone() {
            if inner_node.op != Op::Sub || inner_node.children.len() != 2 {
                continue;
            }
            if egraph.get_constant(inner_node.children[0]).map(|(v, _)| v) != Some(0) {
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

fn apply_constant_folding(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let lhs = snap.children[0];
        let rhs = snap.children[1];
        let Some((lv, lty)) = egraph.get_constant(lhs) else {
            continue;
        };
        let Some((rv, _rty)) = egraph.get_constant(rhs) else {
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
            Op::Shl if rv >= 0 && rv < 64 => Some(lv.wrapping_shl(rv as u32)),
            Op::Shr if rv >= 0 && rv < 64 => Some(((lv as u64).wrapping_shr(rv as u32)) as i64),
            Op::Sar if rv >= 0 && rv < 64 => Some(lv.wrapping_shr(rv as u32)),
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

fn apply_commutativity_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    // Spike test note: this uses canonical-id ordering to ensure each pair is
    // added at most once, preventing combinatorial blowup.  A chain of 10+ Add
    // nodes with commutativity enabled stays well under 50k classes because each
    // Add(a,b) class gains at most one new node (the swapped one), and only when
    // the left child's canonical id is strictly greater than the right child's.
    let mut changed = false;

    for snap in snaps {
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

// ── Reassociation rules ───────────────────────────────────────────────────────
// Add(Add(a, b), c) = Add(a, Add(b, c)) when b or c is constant.
// Mul(Mul(a, b), c) = Mul(a, Mul(b, c)) when b or c is constant.

fn apply_reassociation_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        let is_add = snap.op == Op::Add;
        let is_mul = snap.op == Op::Mul;
        if (!is_add && !is_mul) || snap.children.len() != 2 {
            continue;
        }

        let outer_lhs = snap.children[0];
        let c = snap.children[1];

        let outer_ty = egraph
            .class(egraph.unionfind.find_immutable(class_id))
            .ty
            .clone();

        let inner_canon = egraph.unionfind.find_immutable(outer_lhs);
        let inner_nodes = egraph.class(inner_canon).nodes.clone();

        for inner_node in inner_nodes {
            let inner_matches = if is_add {
                inner_node.op == Op::Add
            } else {
                inner_node.op == Op::Mul
            };
            if !inner_matches || inner_node.children.len() != 2 {
                continue;
            }

            let a = inner_node.children[0];
            let b = inner_node.children[1];

            // All operands must share the same integer type.
            let ty_a = egraph.class(egraph.unionfind.find_immutable(a)).ty.clone();
            let ty_b = egraph.class(egraph.unionfind.find_immutable(b)).ty.clone();
            let ty_c = egraph.class(egraph.unionfind.find_immutable(c)).ty.clone();
            if ty_a != outer_ty || ty_b != outer_ty || ty_c != outer_ty {
                continue;
            }
            if !outer_ty.is_integer() {
                continue;
            }

            // Only reassociate when BOTH b and c are constants AND a is NOT
            // a constant. This targets the pattern (a + 3) + 5 -> a + 8
            // where a is a non-constant expression. If all three are constants,
            // constant folding already handles it. Allowing reassociation on
            // all-constant chains (e.g., 20 chained adds of constants) causes
            // combinatorial blowup via interaction with commutativity.
            let a_is_const = egraph.get_constant(a).is_some();
            let b_is_const = egraph.get_constant(b).is_some();
            let c_is_const = egraph.get_constant(c).is_some();
            if a_is_const || !b_is_const || !c_is_const {
                continue;
            }

            // Build Add/Mul(b, c) then Add/Mul(a, that).
            let inner_op = snap.op.clone();
            let bc = egraph.add(ENode {
                op: inner_op.clone(),
                children: smallvec![b, c],
            });
            let reassoc = egraph.add(ENode {
                op: inner_op,
                children: smallvec![a, bc],
            });

            let orig_canon = egraph.unionfind.find_immutable(class_id);
            let reassoc_canon = egraph.unionfind.find_immutable(reassoc);
            if orig_canon != reassoc_canon {
                egraph.merge(class_id, reassoc);
                changed = true;
            }
        }
    }
    changed
}

// ── Shift combining rules ─────────────────────────────────────────────────────
// Shl(Shl(a, n), m) = Shl(a, n+m) when n, m >= 0 and n+m <= 63.
// Same for Shr and Sar.

fn apply_shift_combining_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        let is_shift = matches!(snap.op, Op::Shl | Op::Shr | Op::Sar);
        if !is_shift || snap.children.len() != 2 {
            continue;
        }

        let outer_val = snap.children[0];
        let m_class = snap.children[1];

        let Some((m, _)) = egraph.get_constant(m_class) else {
            continue;
        };
        if m < 0 {
            continue;
        }

        let outer_canon = egraph.unionfind.find_immutable(outer_val);
        let outer_nodes = egraph.class(outer_canon).nodes.clone();

        for inner_node in outer_nodes {
            if inner_node.op != snap.op || inner_node.children.len() != 2 {
                continue;
            }

            let a = inner_node.children[0];
            let n_class = inner_node.children[1];

            let Some((n, n_ty)) = egraph.get_constant(n_class) else {
                continue;
            };
            if n < 0 {
                continue;
            }
            let combined = n + m;
            // Guard against exceeding the type's bit width (not just 64).
            let class_ty = egraph
                .class(egraph.unionfind.find_immutable(class_id))
                .ty
                .clone();
            let max_shift = if class_ty.is_integer() {
                (class_ty.bit_width() as i64) - 1
            } else {
                63
            };
            if combined > max_shift {
                continue;
            }

            let combined_const = make_iconst(egraph, combined, n_ty);
            let new_shift = egraph.add(ENode {
                op: snap.op.clone(),
                children: smallvec![a, combined_const],
            });

            let orig_canon = egraph.unionfind.find_immutable(class_id);
            let new_canon = egraph.unionfind.find_immutable(new_shift);
            if orig_canon != new_canon {
                egraph.merge(class_id, new_shift);
                changed = true;
            }
        }
    }
    changed
}

// ── Absorption rules ──────────────────────────────────────────────────────────
// And(a, b) where b's class contains Or(a, c) => And class merges with a.
// Or(a, b) where b's class contains And(a, c) => Or class merges with a.
// Both orderings of (a, b) are checked.

fn apply_absorption_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        let is_and = snap.op == Op::And;
        let is_or = snap.op == Op::Or;
        if (!is_and && !is_or) || snap.children.len() != 2 {
            continue;
        }

        // For And(a, b): look for Or(a, _) inside b's class or Or(b, _) inside a's class.
        // For Or(a, b): look for And(a, _) inside b's class or And(b, _) inside a's class.
        let inner_op = if is_and { Op::Or } else { Op::And };

        let children = [snap.children[0], snap.children[1]];

        // Try both orderings: treat children[0] as `a` and children[1] as `b`, then swap.
        for &(a, b) in &[(children[0], children[1]), (children[1], children[0])] {
            let a_canon = egraph.unionfind.find_immutable(a);
            let b_canon = egraph.unionfind.find_immutable(b);
            let b_nodes = egraph.class(b_canon).nodes.clone();

            for inner_node in b_nodes {
                if inner_node.op != inner_op || inner_node.children.len() != 2 {
                    continue;
                }
                let x_canon = egraph.unionfind.find_immutable(inner_node.children[0]);
                let y_canon = egraph.unionfind.find_immutable(inner_node.children[1]);
                if x_canon == a_canon || y_canon == a_canon {
                    let orig_canon = egraph.unionfind.find_immutable(class_id);
                    if orig_canon != a_canon {
                        egraph.merge(class_id, a);
                        changed = true;
                    }
                    break;
                }
            }
        }
    }
    changed
}

// ── Division/remainder identity rules ─────────────────────────────────────────
// SDiv(a, 1) = a, UDiv(a, 1) = a, SRem(a, 1) = 0, URem(a, 1) = 0,
// SDiv(a, -1) = Sub(0, a)

fn apply_div_identity_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let a = snap.children[0];
        let b = snap.children[1];

        match &snap.op {
            Op::SDiv | Op::UDiv => {
                if let Some((1, _)) = egraph.get_constant(b) {
                    // Div(a, 1) = a
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let a_canon = egraph.unionfind.find_immutable(a);
                    if canon != a_canon {
                        egraph.merge(class_id, a);
                        changed = true;
                    }
                } else if snap.op == Op::SDiv {
                    if let Some((-1, _)) = egraph.get_constant(b) {
                        // SDiv(a, -1) = Sub(0, a)
                        let ty = egraph
                            .class(egraph.unionfind.find_immutable(class_id))
                            .ty
                            .clone();
                        let zero = make_iconst(egraph, 0, ty);
                        let neg = egraph.add(ENode {
                            op: Op::Sub,
                            children: smallvec![zero, a],
                        });
                        let canon = egraph.unionfind.find_immutable(class_id);
                        let neg_canon = egraph.unionfind.find_immutable(neg);
                        if canon != neg_canon {
                            egraph.merge(class_id, neg);
                            changed = true;
                        }
                    }
                }
            }
            Op::SRem | Op::URem => {
                if let Some((1, _)) = egraph.get_constant(b) {
                    // Rem(a, 1) = 0
                    let ty = egraph
                        .class(egraph.unionfind.find_immutable(class_id))
                        .ty
                        .clone();
                    let zero = make_iconst(egraph, 0, ty);
                    let canon = egraph.unionfind.find_immutable(class_id);
                    let zero_canon = egraph.unionfind.find_immutable(zero);
                    if canon != zero_canon {
                        egraph.merge(class_id, zero);
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

// ── Select simplification rules ──────────────────────────────────────────────
// Select(c, a, a) = a (same true/false value)
// Select(c, a, b) where c is Icmp on two equal constants -> fold

fn apply_select_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.op != Op::Select || snap.children.len() != 3 {
            continue;
        }
        let t = snap.children[1];
        let f = snap.children[2];

        // Select(c, a, a) = a
        let t_canon = egraph.unionfind.find_immutable(t);
        let f_canon = egraph.unionfind.find_immutable(f);
        if t_canon == f_canon {
            let canon = egraph.unionfind.find_immutable(class_id);
            if canon != t_canon {
                egraph.merge(class_id, t);
                changed = true;
            }
        }
    }
    changed
}

// ── Comparison-select folding ────────────────────────────────────────────────
// Select(Icmp(cc, a, a), t, f) where cc is always-true  -> t
// Select(Icmp(cc, a, a), t, f) where cc is always-false -> f
// Select(Icmp(cc, c1, c2), t, f) where c1,c2 are constants -> t or f

pub(crate) fn eval_icmp(cc: &CondCode, a: i64, b: i64) -> Option<bool> {
    match cc {
        CondCode::Eq => Some(a == b),
        CondCode::Ne => Some(a != b),
        CondCode::Slt => Some(a < b),
        CondCode::Sle => Some(a <= b),
        CondCode::Sgt => Some(a > b),
        CondCode::Sge => Some(a >= b),
        CondCode::Ult => Some((a as u64) < (b as u64)),
        CondCode::Ule => Some((a as u64) <= (b as u64)),
        CondCode::Ugt => Some((a as u64) > (b as u64)),
        CondCode::Uge => Some((a as u64) >= (b as u64)),
        // Float-specific conditions: can't evaluate for integers
        CondCode::Parity | CondCode::NotParity | CondCode::OrdEq | CondCode::UnordNe => None,
    }
}

fn apply_comparison_select_folding(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.op != Op::Select || snap.children.len() != 3 {
            continue;
        }
        let cond_class_id = snap.children[0];
        let t = snap.children[1];
        let f = snap.children[2];

        let cond_canon = egraph.unionfind.find_immutable(cond_class_id);
        let cond_nodes = egraph.class(cond_canon).nodes.clone();

        for icmp_node in &cond_nodes {
            let cc = match &icmp_node.op {
                Op::Icmp(cc) => cc,
                _ => continue,
            };
            if icmp_node.children.len() != 2 {
                continue;
            }

            let a_canon = egraph.unionfind.find_immutable(icmp_node.children[0]);
            let b_canon = egraph.unionfind.find_immutable(icmp_node.children[1]);

            // Pattern A: same-operand comparison
            let target = if a_canon == b_canon {
                let always_true = matches!(
                    cc,
                    CondCode::Eq | CondCode::Sle | CondCode::Sge | CondCode::Ule | CondCode::Uge
                );
                let always_false = matches!(
                    cc,
                    CondCode::Ne | CondCode::Slt | CondCode::Sgt | CondCode::Ult | CondCode::Ugt
                );
                if always_true {
                    Some(t)
                } else if always_false {
                    Some(f)
                } else {
                    None
                }
            } else {
                // Pattern B: constant-constant comparison
                let a_const = egraph.get_constant(icmp_node.children[0]);
                let b_const = egraph.get_constant(icmp_node.children[1]);
                if let (Some((val_a, _)), Some((val_b, _))) = (a_const, b_const) {
                    match eval_icmp(cc, val_a, val_b) {
                        Some(true) => Some(t),
                        Some(false) => Some(f),
                        None => None,
                    }
                } else {
                    None
                }
            };

            if let Some(target) = target {
                let select_canon = egraph.unionfind.find_immutable(class_id);
                let target_canon = egraph.unionfind.find_immutable(target);
                if select_canon != target_canon {
                    egraph.merge(class_id, target);
                    changed = true;
                }
                break;
            }
        }
    }
    changed
}

// ── Extension folding rules ──────────────────────────────────────────────────
// Sext(Sext(a)) = Sext(a) (outer type), Zext(Zext(a)) = Zext(a),
// Trunc(Trunc(a)) = Trunc(a)
// More precisely: the composition of two same-kind extensions is one extension
// from the innermost type to the outermost type.

fn apply_extension_folding_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 1 {
            continue;
        }
        let child = snap.children[0];

        let outer_ty = match &snap.op {
            Op::Sext(ty) | Op::Zext(ty) | Op::Trunc(ty) => ty.clone(),
            _ => continue,
        };

        // Scan child class for a matching inner extension
        let child_canon = egraph.unionfind.find_immutable(child);
        let child_nodes = egraph.class(child_canon).nodes.clone();

        for inner_node in child_nodes {
            if inner_node.children.len() != 1 {
                continue;
            }
            let same_kind = match (&snap.op, &inner_node.op) {
                (Op::Sext(_), Op::Sext(_)) => true,
                (Op::Zext(_), Op::Zext(_)) => true,
                (Op::Trunc(_), Op::Trunc(_)) => true,
                _ => false,
            };
            if !same_kind {
                continue;
            }

            // Ext_outer(Ext_inner(a)) = Ext_outer(a) -- skip the intermediate
            let a = inner_node.children[0];
            let folded_op = match &snap.op {
                Op::Sext(_) => Op::Sext(outer_ty.clone()),
                Op::Zext(_) => Op::Zext(outer_ty.clone()),
                Op::Trunc(_) => Op::Trunc(outer_ty.clone()),
                _ => unreachable!(),
            };
            let folded = egraph.add(ENode {
                op: folded_op,
                children: smallvec![a],
            });
            let canon = egraph.unionfind.find_immutable(class_id);
            let folded_canon = egraph.unionfind.find_immutable(folded);
            if canon != folded_canon {
                egraph.merge(class_id, folded);
                changed = true;
            }
        }
    }
    changed
}

// ── Bitwise complement rules ─────────────────────────────────────────────────
// In this IR, Not(a) is represented as Xor(a, -1).
// Or(a, Xor(a, -1)) = -1, And(a, Xor(a, -1)) = 0

fn apply_complement_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        let is_and = snap.op == Op::And;
        let is_or = snap.op == Op::Or;
        if (!is_and && !is_or) || snap.children.len() != 2 {
            continue;
        }

        let children = [snap.children[0], snap.children[1]];

        // Check both orderings: (a, Xor(a,-1)) and (Xor(a,-1), a)
        for &(a, b) in &[(children[0], children[1]), (children[1], children[0])] {
            let a_canon = egraph.unionfind.find_immutable(a);
            let b_canon = egraph.unionfind.find_immutable(b);
            let b_nodes = egraph.class(b_canon).nodes.clone();

            for b_node in b_nodes {
                if b_node.op != Op::Xor || b_node.children.len() != 2 {
                    continue;
                }
                // Check if Xor(a, -1) or Xor(-1, a)
                let x0 = egraph.unionfind.find_immutable(b_node.children[0]);
                let x1 = egraph.unionfind.find_immutable(b_node.children[1]);
                let is_not_a = (x0 == a_canon
                    && egraph.get_constant(b_node.children[1]).map(|(v, _)| v) == Some(-1))
                    || (x1 == a_canon
                        && egraph.get_constant(b_node.children[0]).map(|(v, _)| v) == Some(-1));
                if !is_not_a {
                    continue;
                }

                let ty = egraph
                    .class(egraph.unionfind.find_immutable(class_id))
                    .ty
                    .clone();
                let result_val = if is_or { -1i64 } else { 0i64 };
                let result = make_iconst(egraph, result_val, ty);
                let canon = egraph.unionfind.find_immutable(class_id);
                let result_canon = egraph.unionfind.find_immutable(result);
                if canon != result_canon {
                    egraph.merge(class_id, result);
                    changed = true;
                }
                break;
            }
        }
    }
    changed
}

// ── De Morgan's law rules ────────────────────────────────────────────────────
// Not(And(a,b)) = Or(Not(a), Not(b))
// Not(Or(a,b)) = And(Not(a), Not(b))
// Where Not(x) = Xor(x, -1).

fn apply_demorgan_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        // Match Xor(inner, -1) where inner is And or Or
        if snap.op != Op::Xor || snap.children.len() != 2 {
            continue;
        }

        // Identify which child is -1 and which is the inner op
        let (inner, neg_child) = {
            let c0 = snap.children[0];
            let c1 = snap.children[1];
            if egraph.get_constant(c1).map(|(v, _)| v) == Some(-1) {
                (c0, c1)
            } else if egraph.get_constant(c0).map(|(v, _)| v) == Some(-1) {
                (c1, c0)
            } else {
                continue;
            }
        };

        let inner_canon = egraph.unionfind.find_immutable(inner);
        let inner_nodes = egraph.class(inner_canon).nodes.clone();

        for inner_node in inner_nodes {
            if inner_node.children.len() != 2 {
                continue;
            }
            let (target_op, inner_is_and_or_or) = match &inner_node.op {
                Op::And => (Op::Or, true),
                Op::Or => (Op::And, true),
                _ => (Op::And, false), // placeholder
            };
            if !inner_is_and_or_or {
                continue;
            }

            let a = inner_node.children[0];
            let b = inner_node.children[1];

            // Build Not(a) = Xor(a, -1), Not(b) = Xor(b, -1)
            let not_a = egraph.add(ENode {
                op: Op::Xor,
                children: smallvec![a, neg_child],
            });
            let not_b = egraph.add(ENode {
                op: Op::Xor,
                children: smallvec![b, neg_child],
            });
            // Build target_op(Not(a), Not(b))
            let result = egraph.add(ENode {
                op: target_op,
                children: smallvec![not_a, not_b],
            });

            let canon = egraph.unionfind.find_immutable(class_id);
            let result_canon = egraph.unionfind.find_immutable(result);
            if canon != result_canon {
                egraph.merge(class_id, result);
                changed = true;
            }
        }
    }
    changed
}

// ── Negation distribution rule ───────────────────────────────────────────────
// Sub(0, Add(a, b)) = Add(Sub(0, a), Sub(0, b))

fn apply_negation_distribution_rules(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        // Match Sub(0, inner) where inner is Add
        if snap.op != Op::Sub || snap.children.len() != 2 {
            continue;
        }
        if egraph.get_constant(snap.children[0]).map(|(v, _)| v) != Some(0) {
            continue;
        }

        let inner = snap.children[1];
        let zero = snap.children[0];
        let inner_canon = egraph.unionfind.find_immutable(inner);
        let inner_nodes = egraph.class(inner_canon).nodes.clone();

        for inner_node in inner_nodes {
            if inner_node.op != Op::Add || inner_node.children.len() != 2 {
                continue;
            }
            let a = inner_node.children[0];
            let b = inner_node.children[1];

            // Build Sub(0, a), Sub(0, b), Add(Sub(0,a), Sub(0,b))
            let neg_a = egraph.add(ENode {
                op: Op::Sub,
                children: smallvec![zero, a],
            });
            let neg_b = egraph.add(ENode {
                op: Op::Sub,
                children: smallvec![zero, b],
            });
            let result = egraph.add(ENode {
                op: Op::Add,
                children: smallvec![neg_a, neg_b],
            });

            let canon = egraph.unionfind.find_immutable(class_id);
            let result_canon = egraph.unionfind.find_immutable(result);
            if canon != result_canon {
                egraph.merge(class_id, result);
                changed = true;
            }
        }
    }
    changed
}

// ── Block-param constant propagation ─────────────────────────────────────────

/// For blocks with a single predecessor, merge each block parameter's e-class
/// with the corresponding argument e-class from that predecessor. This enables
/// constant folding through inlined function boundaries.
pub fn propagate_block_params(func: &Function, egraph: &mut EGraph) {
    // Step 1: Build predecessor map: block -> vec of (source_block, args).
    let mut pred_map: BTreeMap<BlockId, Vec<(BlockId, Vec<ClassId>)>> = BTreeMap::new();
    for block in &func.blocks {
        for op in &block.ops {
            match op {
                EffectfulOp::Jump { target, args } => {
                    pred_map
                        .entry(*target)
                        .or_default()
                        .push((block.id, args.clone()));
                }
                EffectfulOp::Branch {
                    bb_true,
                    bb_false,
                    true_args,
                    false_args,
                    ..
                } => {
                    pred_map
                        .entry(*bb_true)
                        .or_default()
                        .push((block.id, true_args.clone()));
                    pred_map
                        .entry(*bb_false)
                        .or_default()
                        .push((block.id, false_args.clone()));
                }
                _ => {}
            }
        }
    }

    // Step 2: Build block_param lookup: (block_id, param_index) -> ClassId.
    let mut block_param_map: BTreeMap<(BlockId, u32), ClassId> = BTreeMap::new();
    for i in 0..egraph.classes.len() as u32 {
        let id = ClassId(i);
        if egraph.unionfind.find_immutable(id) != id {
            continue;
        }
        let class = egraph.class(id);
        for node in &class.nodes {
            if let Op::BlockParam(block_id, param_idx, _) = &node.op {
                block_param_map.insert((*block_id, *param_idx), id);
            }
        }
    }

    // Step 3: For single-predecessor blocks, merge block params with constant args.
    // Only merge when the source arg contains a constant, since merging with
    // non-constant values can cause extraction to schedule computations in the
    // wrong block (the source computation may not dominate the target block).
    let mut merged = false;
    for block in &func.blocks {
        if block.param_types.is_empty() {
            continue;
        }
        let preds = match pred_map.get(&block.id) {
            Some(p) if p.len() == 1 => p,
            _ => continue,
        };
        let (_, ref args) = preds[0];
        for i in 0..block.param_types.len() {
            let Some(&bp_class) = block_param_map.get(&(block.id, i as u32)) else {
                continue;
            };
            let source_class = args[i];
            // Only propagate constants to avoid extraction scheduling issues.
            if egraph.get_constant(source_class).is_none() {
                continue;
            }
            let bp_canon = egraph.unionfind.find(bp_class);
            let src_canon = egraph.unionfind.find(source_class);
            if bp_canon != src_canon {
                egraph.merge(bp_class, source_class);
                merged = true;
            }
        }
    }

    // Step 4: Rebuild if any merges happened.
    if merged {
        egraph.rebuild();
    }
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
        let snaps = snapshot_all(&g);
        apply_commutativity_rules(&mut g, &snaps);
        g.rebuild();
        let count = g.class_count();
        assert!(
            count < 50_000,
            "e-class count {count} exceeded blowup threshold"
        );
    }

    // Reassociation: (a + 3) + 5 should fold to a + 8
    #[test]
    fn reassociate_add_constants() {
        use crate::egraph::phases::{CompileOptions, run_phases};
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let c3 = iconst(&mut g, 3, Type::I64);
        let c5 = iconst(&mut g, 5, Type::I64);
        let inner = add(&mut g, a, c3);
        let outer = add(&mut g, inner, c5);
        run_phases(&mut g, &CompileOptions::default()).unwrap();
        // After reassociation + constant folding: (a+3)+5 = a+(3+5) = a+8
        let c8 = iconst(&mut g, 8, Type::I64);
        let expected = add(&mut g, a, c8);
        assert_eq!(g.find(outer), g.find(expected));
    }

    // Shift combining: Shl(Shl(a, 2), 3) = Shl(a, 5)
    #[test]
    fn shift_combine_shl() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let c2 = iconst(&mut g, 2, Type::I64);
        let c3 = iconst(&mut g, 3, Type::I64);
        let inner = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, c2],
        });
        let outer = g.add(ENode {
            op: Op::Shl,
            children: smallvec![inner, c3],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let c5 = iconst(&mut g, 5, Type::I64);
        let combined = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, c5],
        });
        assert_eq!(g.find(outer), g.find(combined));
    }

    // Shift combining overflow guard: Shl(Shl(a, 32), 32) should NOT combine
    #[test]
    fn shift_combine_overflow_guard() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1, Type::I64);
        let c32 = iconst(&mut g, 32, Type::I64);
        let inner = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, c32],
        });
        let outer = g.add(ENode {
            op: Op::Shl,
            children: smallvec![inner, c32],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let c64 = iconst(&mut g, 64, Type::I64);
        let bad_combined = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, c64],
        });
        // Should NOT be merged (64 > 63)
        assert_ne!(g.find(outer), g.find(bad_combined));
    }

    // Absorption: And(a, Or(a, b)) = a
    #[test]
    fn absorption_and_or() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 0xFF, Type::I64);
        let b = iconst(&mut g, 0x0F, Type::I64);
        let or_ab = g.add(ENode {
            op: Op::Or,
            children: smallvec![a, b],
        });
        let and_result = g.add(ENode {
            op: Op::And,
            children: smallvec![a, or_ab],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(and_result), g.find(a));
    }

    // Absorption: Or(a, And(a, b)) = a
    #[test]
    fn absorption_or_and() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 0xFF, Type::I64);
        let b = iconst(&mut g, 0x0F, Type::I64);
        let and_ab = g.add(ENode {
            op: Op::And,
            children: smallvec![a, b],
        });
        let or_result = g.add(ENode {
            op: Op::Or,
            children: smallvec![a, and_ab],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(or_result), g.find(a));
    }

    // ── Division/remainder identity tests ────────────────────────────────────

    #[test]
    fn sdiv_by_one() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let one = iconst(&mut g, 1, Type::I64);
        let div = g.add(ENode {
            op: Op::SDiv,
            children: smallvec![a, one],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(div), g.find(a));
    }

    #[test]
    fn udiv_by_one() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let one = iconst(&mut g, 1, Type::I64);
        let div = g.add(ENode {
            op: Op::UDiv,
            children: smallvec![a, one],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(div), g.find(a));
    }

    #[test]
    fn srem_by_one() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let one = iconst(&mut g, 1, Type::I64);
        let rem = g.add(ENode {
            op: Op::SRem,
            children: smallvec![a, one],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let zero = iconst(&mut g, 0, Type::I64);
        assert_eq!(g.find(rem), g.find(zero));
    }

    #[test]
    fn urem_by_one() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let one = iconst(&mut g, 1, Type::I64);
        let rem = g.add(ENode {
            op: Op::URem,
            children: smallvec![a, one],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let zero = iconst(&mut g, 0, Type::I64);
        assert_eq!(g.find(rem), g.find(zero));
    }

    #[test]
    fn sdiv_by_neg_one() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let neg1 = iconst(&mut g, -1, Type::I64);
        let div = g.add(ENode {
            op: Op::SDiv,
            children: smallvec![a, neg1],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // SDiv(a, -1) = Sub(0, a) = -42
        let zero = iconst(&mut g, 0, Type::I64);
        let neg = g.add(ENode {
            op: Op::Sub,
            children: smallvec![zero, a],
        });
        assert_eq!(g.find(div), g.find(neg));
    }

    // ── Select simplification tests ──────────────────────────────────────────

    #[test]
    fn select_same_branches() {
        let mut g = EGraph::new();
        let c1 = iconst(&mut g, 1, Type::I64);
        let c2 = iconst(&mut g, 2, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(crate::ir::condcode::CondCode::Eq),
            children: smallvec![c1, c2],
        });
        let val = iconst(&mut g, 99, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, val, val],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(sel), g.find(val));
    }

    // ── Comparison-select folding tests ─────────────────────────────────────

    #[test]
    fn icmp_eq_same_operand_folds_select() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 7, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Eq),
            children: smallvec![a, a],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Eq(a, a) is always true: Select folds to t
        assert_eq!(g.find(sel), g.find(t));
    }

    #[test]
    fn icmp_ne_same_operand_folds_select() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 7, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Ne),
            children: smallvec![a, a],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Ne(a, a) is always false: Select folds to f
        assert_eq!(g.find(sel), g.find(f));
    }

    #[test]
    fn icmp_slt_same_operand_folds_select() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 7, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Slt),
            children: smallvec![a, a],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Slt(a, a) is always false: Select folds to f
        assert_eq!(g.find(sel), g.find(f));
    }

    #[test]
    fn icmp_const_const_slt_true() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 3, Type::I64);
        let b = iconst(&mut g, 5, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Slt),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // 3 < 5 is true
        assert_eq!(g.find(sel), g.find(t));
    }

    #[test]
    fn icmp_const_const_slt_false() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 5, Type::I64);
        let b = iconst(&mut g, 3, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Slt),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // 5 < 3 is false
        assert_eq!(g.find(sel), g.find(f));
    }

    #[test]
    fn icmp_const_const_ult_unsigned() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, -1, Type::I64);
        let b = iconst(&mut g, 1, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Ult),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // -1 as u64 is u64::MAX, which is not < 1, so this is false
        assert_eq!(g.find(sel), g.find(f));
    }

    #[test]
    fn icmp_const_const_eq_true() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let b = iconst(&mut g, 42, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Eq),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // 42 == 42 is true
        assert_eq!(g.find(sel), g.find(t));
    }

    #[test]
    fn icmp_sle_same_operand_folds_select() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 7, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Sle),
            children: smallvec![a, a],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Sle(a, a) is always true
        assert_eq!(g.find(sel), g.find(t));
    }

    #[test]
    fn icmp_uge_same_operand_folds_select() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 7, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Uge),
            children: smallvec![a, a],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Uge(a, a) is always true
        assert_eq!(g.find(sel), g.find(t));
    }

    #[test]
    fn icmp_ugt_same_operand_folds_select() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 7, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Ugt),
            children: smallvec![a, a],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Ugt(a, a) is always false
        assert_eq!(g.find(sel), g.find(f));
    }

    #[test]
    fn icmp_const_const_sge() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 5, Type::I64);
        let b = iconst(&mut g, 3, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Sge),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // 5 >= 3 is true
        assert_eq!(g.find(sel), g.find(t));
    }

    #[test]
    fn icmp_const_const_ne_false() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 7, Type::I64);
        let b = iconst(&mut g, 7, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Ne),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // 7 != 7 is false
        assert_eq!(g.find(sel), g.find(f));
    }

    #[test]
    fn icmp_const_const_ule() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 3, Type::I64);
        let b = iconst(&mut g, 3, Type::I64);
        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Ule),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1, Type::I64);
        let f = iconst(&mut g, 0, Type::I64);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, t, f],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // 3 <= 3 (unsigned) is true
        assert_eq!(g.find(sel), g.find(t));
    }

    // ── Extension folding tests ──────────────────────────────────────────────

    #[test]
    fn sext_sext_folds() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 5, Type::I8);
        let inner = g.add(ENode {
            op: Op::Sext(Type::I32),
            children: smallvec![a],
        });
        let outer = g.add(ENode {
            op: Op::Sext(Type::I64),
            children: smallvec![inner],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Sext(I64, Sext(I32, a)) = Sext(I64, a)
        let direct = g.add(ENode {
            op: Op::Sext(Type::I64),
            children: smallvec![a],
        });
        assert_eq!(g.find(outer), g.find(direct));
    }

    #[test]
    fn zext_zext_folds() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 5, Type::I8);
        let inner = g.add(ENode {
            op: Op::Zext(Type::I32),
            children: smallvec![a],
        });
        let outer = g.add(ENode {
            op: Op::Zext(Type::I64),
            children: smallvec![inner],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let direct = g.add(ENode {
            op: Op::Zext(Type::I64),
            children: smallvec![a],
        });
        assert_eq!(g.find(outer), g.find(direct));
    }

    #[test]
    fn trunc_trunc_folds() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1000, Type::I64);
        let inner = g.add(ENode {
            op: Op::Trunc(Type::I32),
            children: smallvec![a],
        });
        let outer = g.add(ENode {
            op: Op::Trunc(Type::I8),
            children: smallvec![inner],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let direct = g.add(ENode {
            op: Op::Trunc(Type::I8),
            children: smallvec![a],
        });
        assert_eq!(g.find(outer), g.find(direct));
    }

    // ── Bitwise complement tests ─────────────────────────────────────────────

    #[test]
    fn or_a_not_a_is_all_ones() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 0xFF, Type::I64);
        let neg1 = iconst(&mut g, -1, Type::I64);
        let not_a = g.add(ENode {
            op: Op::Xor,
            children: smallvec![a, neg1],
        });
        let result = g.add(ENode {
            op: Op::Or,
            children: smallvec![a, not_a],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(result), g.find(neg1));
    }

    #[test]
    fn and_a_not_a_is_zero() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 0xFF, Type::I64);
        let neg1 = iconst(&mut g, -1, Type::I64);
        let not_a = g.add(ENode {
            op: Op::Xor,
            children: smallvec![a, neg1],
        });
        let result = g.add(ENode {
            op: Op::And,
            children: smallvec![a, not_a],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let zero = iconst(&mut g, 0, Type::I64);
        assert_eq!(g.find(result), g.find(zero));
    }

    // ── Or annihilation test ─────────────────────────────────────────────────

    #[test]
    fn or_a_all_ones() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 42, Type::I64);
        let neg1 = iconst(&mut g, -1, Type::I64);
        let result = g.add(ENode {
            op: Op::Or,
            children: smallvec![a, neg1],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        assert_eq!(g.find(result), g.find(neg1));
    }

    // ── De Morgan's law tests ────────────────────────────────────────────────

    #[test]
    fn demorgan_not_and() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 0xF0, Type::I64);
        let b = iconst(&mut g, 0x0F, Type::I64);
        let neg1 = iconst(&mut g, -1, Type::I64);
        let and_ab = g.add(ENode {
            op: Op::And,
            children: smallvec![a, b],
        });
        // Not(And(a, b)) = Xor(And(a, b), -1)
        let not_and = g.add(ENode {
            op: Op::Xor,
            children: smallvec![and_ab, neg1],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Should equal Or(Xor(a, -1), Xor(b, -1))
        let not_a = g.add(ENode {
            op: Op::Xor,
            children: smallvec![a, neg1],
        });
        let not_b = g.add(ENode {
            op: Op::Xor,
            children: smallvec![b, neg1],
        });
        let or_not = g.add(ENode {
            op: Op::Or,
            children: smallvec![not_a, not_b],
        });
        assert_eq!(g.find(not_and), g.find(or_not));
    }

    #[test]
    fn demorgan_not_or() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 0xF0, Type::I64);
        let b = iconst(&mut g, 0x0F, Type::I64);
        let neg1 = iconst(&mut g, -1, Type::I64);
        let or_ab = g.add(ENode {
            op: Op::Or,
            children: smallvec![a, b],
        });
        let not_or = g.add(ENode {
            op: Op::Xor,
            children: smallvec![or_ab, neg1],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        let not_a = g.add(ENode {
            op: Op::Xor,
            children: smallvec![a, neg1],
        });
        let not_b = g.add(ENode {
            op: Op::Xor,
            children: smallvec![b, neg1],
        });
        let and_not = g.add(ENode {
            op: Op::And,
            children: smallvec![not_a, not_b],
        });
        assert_eq!(g.find(not_or), g.find(and_not));
    }

    // ── Icmp(Eq/Ne, Sub(a,b), 0) rewrite tests ───────────────────────────────

    #[test]
    fn icmp_eq_sub_zero_rewrites_to_icmp_eq_ab() {
        use crate::ir::condcode::CondCode;
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let b = g.add(ENode {
            op: Op::Param(1, Type::I64),
            children: smallvec![],
        });
        let zero = iconst(&mut g, 0, Type::I64);
        let diff = g.add(ENode {
            op: Op::Sub,
            children: smallvec![a, b],
        });
        let eq = g.add(ENode {
            op: Op::Icmp(CondCode::Eq),
            children: smallvec![diff, zero],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();

        let direct = g.add(ENode {
            op: Op::Icmp(CondCode::Eq),
            children: smallvec![a, b],
        });
        assert_eq!(g.find(eq), g.find(direct));
    }

    #[test]
    fn icmp_ne_zero_sub_symmetric() {
        use crate::ir::condcode::CondCode;
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Param(0, Type::I32),
            children: smallvec![],
        });
        let b = g.add(ENode {
            op: Op::Param(1, Type::I32),
            children: smallvec![],
        });
        let zero = iconst(&mut g, 0, Type::I32);
        let diff = g.add(ENode {
            op: Op::Sub,
            children: smallvec![a, b],
        });
        // RHS-first form: Icmp(Ne, 0, Sub(a, b))
        let ne = g.add(ENode {
            op: Op::Icmp(CondCode::Ne),
            children: smallvec![zero, diff],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();

        let direct = g.add(ENode {
            op: Op::Icmp(CondCode::Ne),
            children: smallvec![a, b],
        });
        assert_eq!(g.find(ne), g.find(direct));
    }

    #[test]
    fn icmp_sgt_sub_zero_not_rewritten() {
        // Signed ordering across Sub's overflow is NOT equivalent: leave alone.
        use crate::ir::condcode::CondCode;
        let mut g = EGraph::new();
        let a = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let b = g.add(ENode {
            op: Op::Param(1, Type::I64),
            children: smallvec![],
        });
        let zero = iconst(&mut g, 0, Type::I64);
        let diff = g.add(ENode {
            op: Op::Sub,
            children: smallvec![a, b],
        });
        let sgt = g.add(ENode {
            op: Op::Icmp(CondCode::Sgt),
            children: smallvec![diff, zero],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();

        let direct = g.add(ENode {
            op: Op::Icmp(CondCode::Sgt),
            children: smallvec![a, b],
        });
        // Must NOT be merged.
        assert_ne!(g.find(sgt), g.find(direct));
    }

    // ── Negation distribution test ───────────────────────────────────────────

    #[test]
    fn neg_distributes_over_add() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 3, Type::I64);
        let b = iconst(&mut g, 5, Type::I64);
        let zero = iconst(&mut g, 0, Type::I64);
        let sum = add(&mut g, a, b);
        let neg_sum = g.add(ENode {
            op: Op::Sub,
            children: smallvec![zero, sum],
        });
        apply_algebraic_rules(&mut g);
        g.rebuild();
        // Sub(0, Add(a, b)) = Add(Sub(0, a), Sub(0, b))
        let neg_a = g.add(ENode {
            op: Op::Sub,
            children: smallvec![zero, a],
        });
        let neg_b = g.add(ENode {
            op: Op::Sub,
            children: smallvec![zero, b],
        });
        let distributed = add(&mut g, neg_a, neg_b);
        assert_eq!(g.find(neg_sum), g.find(distributed));
    }
}
