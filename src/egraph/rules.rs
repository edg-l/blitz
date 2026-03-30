use std::collections::HashMap;

use crate::egraph::egraph::EGraph;
use crate::egraph::enode::ENode;
use crate::ir::op::{ClassId, Op};

pub type VarId = u32;

/// A pattern for matching or building e-graph nodes.
pub enum Pattern {
    /// Match any e-class; bind the matched ClassId to this variable.
    Var(VarId),
    /// Match a specific op with sub-patterns for each child.
    Op(Op, Vec<Pattern>),
    /// Match an Iconst whose value satisfies a predicate.
    ConstPred(Box<dyn Fn(i64) -> bool>),
}

/// Variable bindings from a pattern match.
pub type Substitution = HashMap<VarId, ClassId>;

/// A rewrite rule is a function that inspects the e-graph and may add equivalences.
/// Returns true if any changes were made.
pub type RewriteRule = fn(&mut EGraph) -> bool;

/// Search all nodes in `class_id` for matches against `pattern`.
/// Returns all valid substitutions (there may be multiple).
pub fn match_pattern(egraph: &EGraph, pattern: &Pattern, class_id: ClassId) -> Vec<Substitution> {
    let canon = egraph.unionfind.find_immutable(class_id);
    if canon == ClassId::NONE {
        return vec![];
    }
    let class = egraph.class(canon);

    match pattern {
        Pattern::Var(v) => {
            let mut subst = Substitution::new();
            subst.insert(*v, canon);
            vec![subst]
        }

        Pattern::ConstPred(pred) => {
            let mut results = vec![];
            for node in &class.nodes {
                if let Op::Iconst(val, _) = &node.op {
                    if pred(*val) {
                        results.push(Substitution::new());
                    }
                }
            }
            results
        }

        Pattern::Op(op, child_pats) => {
            let mut results = vec![];
            for node in &class.nodes {
                if &node.op != op {
                    continue;
                }
                if node.children.len() != child_pats.len() {
                    continue;
                }
                // Recursively match each child pattern
                let mut subst_sets: Vec<Vec<Substitution>> = Vec::new();
                let mut ok = true;
                for (child_id, child_pat) in node.children.iter().zip(child_pats.iter()) {
                    let child_matches = match_pattern(egraph, child_pat, *child_id);
                    if child_matches.is_empty() {
                        ok = false;
                        break;
                    }
                    subst_sets.push(child_matches);
                }
                if !ok {
                    continue;
                }
                // Combine substitutions via cross-product, checking consistency
                let combined = combine_substitutions(&subst_sets);
                results.extend(combined);
            }
            results
        }
    }
}

/// Combine multiple sets of substitutions via cross-product, merging consistent bindings.
fn combine_substitutions(sets: &[Vec<Substitution>]) -> Vec<Substitution> {
    if sets.is_empty() {
        return vec![Substitution::new()];
    }
    let mut acc = sets[0].clone();
    for set in &sets[1..] {
        let mut next = vec![];
        for left in &acc {
            for right in set {
                if let Some(merged) = merge_subst(left, right) {
                    next.push(merged);
                }
            }
        }
        acc = next;
    }
    acc
}

/// Merge two substitutions; returns None if they conflict.
fn merge_subst(a: &Substitution, b: &Substitution) -> Option<Substitution> {
    let mut result = a.clone();
    for (var, class) in b {
        if let Some(existing) = result.get(var) {
            if existing != class {
                return None;
            }
        } else {
            result.insert(*var, *class);
        }
    }
    Some(result)
}

/// Build new e-graph nodes from a pattern + substitution; returns the ClassId of the root.
pub fn instantiate(egraph: &mut EGraph, pattern: &Pattern, subst: &Substitution) -> ClassId {
    match pattern {
        Pattern::Var(v) => *subst
            .get(v)
            .unwrap_or_else(|| panic!("variable {v} not bound in substitution")),

        Pattern::ConstPred(_) => panic!("ConstPred cannot be used on the RHS of a rule"),

        Pattern::Op(op, child_pats) => {
            let children: smallvec::SmallVec<[ClassId; 2]> = child_pats
                .iter()
                .map(|p| instantiate(egraph, p, subst))
                .collect();
            egraph.add(ENode {
                op: op.clone(),
                children,
            })
        }
    }
}

/// Apply a rewrite rule: iterate all classes, match LHS, instantiate RHS, merge.
/// Returns true if any new equivalences were added.
pub fn apply_rule(egraph: &mut EGraph, lhs: &Pattern, rhs: &Pattern) -> bool {
    // Snapshot class IDs first to avoid mutation during iteration
    let class_ids: Vec<ClassId> = (0..egraph.classes.len() as u32)
        .map(ClassId)
        .filter(|&id| egraph.unionfind.find_immutable(id) == id)
        .collect();

    let mut changed = false;
    for class_id in class_ids {
        let matches = match_pattern(egraph, lhs, class_id);
        for subst in matches {
            let rhs_class = instantiate(egraph, rhs, &subst);
            let canon = egraph.unionfind.find_immutable(class_id);
            if egraph.unionfind.find_immutable(rhs_class) != canon {
                egraph.merge(class_id, rhs_class);
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::ir::types::Type;

    fn iconst(g: &mut EGraph, v: i64) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(v, Type::I64),
            children: smallvec![],
        })
    }

    // 4.4: simple var match
    #[test]
    fn var_matches_any_class() {
        let mut g = EGraph::new();
        let c = iconst(&mut g, 42);
        let pat = Pattern::Var(0);
        let matches = match_pattern(&g, &pat, c);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0][&0], c);
    }

    // 4.4: nested op match
    #[test]
    fn op_pattern_matches_add() {
        let mut g = EGraph::new();
        let c1 = iconst(&mut g, 1);
        let c2 = iconst(&mut g, 2);
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![c1, c2],
        });

        // Pattern: Add(?0, ?1)
        let pat = Pattern::Op(Op::Add, vec![Pattern::Var(0), Pattern::Var(1)]);
        let matches = match_pattern(&g, &pat, add);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0][&0], c1);
        assert_eq!(matches[0][&1], c2);
    }

    // 4.4: const predicate match
    #[test]
    fn const_pred_matches_zero() {
        let mut g = EGraph::new();
        let c = iconst(&mut g, 0);
        let pat = Pattern::ConstPred(Box::new(|v| v == 0));
        let matches = match_pattern(&g, &pat, c);
        assert_eq!(matches.len(), 1);
    }

    // 4.4: const predicate no match
    #[test]
    fn const_pred_no_match() {
        let mut g = EGraph::new();
        let c = iconst(&mut g, 5);
        let pat = Pattern::ConstPred(Box::new(|v| v == 0));
        let matches = match_pattern(&g, &pat, c);
        assert!(matches.is_empty());
    }

    // 4.5: no match returns empty
    #[test]
    fn op_pattern_no_match() {
        let mut g = EGraph::new();
        let c = iconst(&mut g, 1);
        // Pattern expects Add, but c is Iconst
        let pat = Pattern::Op(Op::Add, vec![Pattern::Var(0), Pattern::Var(1)]);
        let matches = match_pattern(&g, &pat, c);
        assert!(matches.is_empty());
    }

    // 4.5: nested match
    #[test]
    fn nested_pattern_match() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 10);
        let two = iconst(&mut g, 2);
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, two],
        });
        let b = iconst(&mut g, 5);
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![b, shl],
        });

        // Pattern: Add(?0, Shl(?1, ConstPred(==2)))
        let pat = Pattern::Op(
            Op::Add,
            vec![
                Pattern::Var(0),
                Pattern::Op(
                    Op::Shl,
                    vec![Pattern::Var(1), Pattern::ConstPred(Box::new(|v| v == 2))],
                ),
            ],
        );
        let matches = match_pattern(&g, &pat, add);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0][&0], b);
        assert_eq!(matches[0][&1], a);
    }
}
