use smallvec::smallvec;

use crate::egraph::algebraic::find_iconst;
use crate::egraph::egraph::{EGraph, snapshot_all};
use crate::egraph::enode::ENode;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

pub fn apply_addr_mode_rules(egraph: &mut EGraph) -> bool {
    let mut changed = false;
    changed |= apply_addr_rules(egraph);
    changed |= apply_lea_rules(egraph);
    changed
}

/// Valid x86-64 scales.
fn is_valid_scale(s: u8) -> bool {
    matches!(s, 1 | 2 | 4 | 8)
}

// ── Addr formation rules ──────────────────────────────────────────────────────

fn apply_addr_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        if snap.op != Op::Add || snap.children.len() != 2 {
            continue;
        }

        // Addr formation only applies to I64 pointer arithmetic.
        let class_ty = egraph
            .class(egraph.unionfind.find_immutable(class_id))
            .ty
            .clone();
        if class_ty != Type::I64 {
            continue;
        }

        let base = snap.children[0];
        let rhs = snap.children[1];

        // Pattern 1: Add(base, Iconst(d)) -> Addr{scale:1, disp:d}(base, NONE)
        //   where d fits i32
        if let Some((d, _)) = find_iconst(egraph, rhs) {
            if d >= i32::MIN as i64 && d <= i32::MAX as i64 {
                let addr = egraph.add(ENode {
                    op: Op::Addr {
                        scale: 1,
                        disp: d as i32,
                    },
                    children: smallvec![base, ClassId::NONE],
                });
                let canon = egraph.unionfind.find_immutable(class_id);
                if egraph.unionfind.find_immutable(addr) != canon {
                    egraph.merge(class_id, addr);
                    changed = true;
                }
            }
        }

        // Pattern 2: Add(base, Shl(idx, Iconst(n))) -> Addr{scale:2^n, disp:0}(base, idx)
        //   for n in {1,2,3}
        {
            let rhs_canon = egraph.unionfind.find_immutable(rhs);
            let rhs_class = egraph.class(rhs_canon);
            let rhs_nodes: Vec<_> = rhs_class.nodes.clone();
            for rhs_node in &rhs_nodes {
                if rhs_node.op == Op::Shl && rhs_node.children.len() == 2 {
                    let idx = rhs_node.children[0];
                    let shift = rhs_node.children[1];
                    if let Some((n, _)) = find_iconst(egraph, shift) {
                        if matches!(n, 1 | 2 | 3) {
                            let scale = 1u8 << n;
                            let addr = egraph.add(ENode {
                                op: Op::Addr { scale, disp: 0 },
                                children: smallvec![base, idx],
                            });
                            let canon = egraph.unionfind.find_immutable(class_id);
                            if egraph.unionfind.find_immutable(addr) != canon {
                                egraph.merge(class_id, addr);
                                changed = true;
                            }
                        }
                    }
                }
                // Pattern: Add(base, Mul(idx, Iconst(s))) -> Addr{scale:s, disp:0}(base, idx)
                //   for s in {2,4,8}
                if rhs_node.op == Op::Mul && rhs_node.children.len() == 2 {
                    let [mc0, mc1] = [rhs_node.children[0], rhs_node.children[1]];
                    let (scale_opt, idx) = if let Some((s, _)) = find_iconst(egraph, mc1) {
                        if s > 0 && s <= 8 {
                            (Some(s as u8), mc0)
                        } else {
                            (None, mc0)
                        }
                    } else if let Some((s, _)) = find_iconst(egraph, mc0) {
                        if s > 0 && s <= 8 {
                            (Some(s as u8), mc1)
                        } else {
                            (None, mc1)
                        }
                    } else {
                        (None, mc0)
                    };
                    if let Some(s) = scale_opt {
                        if is_valid_scale(s) && s != 1 {
                            let addr = egraph.add(ENode {
                                op: Op::Addr { scale: s, disp: 0 },
                                children: smallvec![base, idx],
                            });
                            let canon = egraph.unionfind.find_immutable(class_id);
                            if egraph.unionfind.find_immutable(addr) != canon {
                                egraph.merge(class_id, addr);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Pattern 3: Add(base, idx) -> Addr{scale:1, disp:0}(base, idx) [generic]
        {
            let addr = egraph.add(ENode {
                op: Op::Addr { scale: 1, disp: 0 },
                children: smallvec![base, rhs],
            });
            let canon = egraph.unionfind.find_immutable(class_id);
            if egraph.unionfind.find_immutable(addr) != canon {
                egraph.merge(class_id, addr);
                changed = true;
            }
        }
    }
    changed
}

// ── LEA formation rules ───────────────────────────────────────────────────────

fn apply_lea_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;

        match &snap.op {
            // Add(a, b) -> X86Lea2(a, b)
            Op::Add if snap.children.len() == 2 => {
                let a = snap.children[0];
                let b = snap.children[1];
                // Only apply LEA to I64 operands
                let a_ty = egraph.class(egraph.unionfind.find_immutable(a)).ty.clone();
                let b_ty = egraph.class(egraph.unionfind.find_immutable(b)).ty.clone();
                if a_ty == Type::I64 && b_ty == Type::I64 {
                    let lea2 = egraph.add(ENode {
                        op: Op::X86Lea2,
                        children: smallvec![a, b],
                    });
                    let canon = egraph.unionfind.find_immutable(class_id);
                    if egraph.unionfind.find_immutable(lea2) != canon {
                        egraph.merge(class_id, lea2);
                        changed = true;
                    }
                }

                // Add(a, Shl(b, Iconst(n))) -> X86Lea3(a, b, 2^n) for n in {1,2,3}
                let rhs_canon = egraph.unionfind.find_immutable(b);
                let rhs_nodes: Vec<_> = egraph.class(rhs_canon).nodes.clone();
                for rhs_node in &rhs_nodes {
                    if rhs_node.op == Op::Shl && rhs_node.children.len() == 2 {
                        let idx = rhs_node.children[0];
                        let shift = rhs_node.children[1];
                        if let Some((n, _)) = find_iconst(egraph, shift) {
                            if matches!(n, 1 | 2 | 3) {
                                let scale = 1u8 << n;
                                let idx_ty = egraph
                                    .class(egraph.unionfind.find_immutable(idx))
                                    .ty
                                    .clone();
                                if a_ty == Type::I64 && idx_ty == Type::I64 {
                                    let lea3 = egraph.add(ENode {
                                        op: Op::X86Lea3 { scale },
                                        children: smallvec![a, idx],
                                    });
                                    let canon = egraph.unionfind.find_immutable(class_id);
                                    if egraph.unionfind.find_immutable(lea3) != canon {
                                        egraph.merge(class_id, lea3);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // Add(a, Iconst(d)) -> X86Lea4(a, NONE, 1, d)
                if let Some((d, _)) = find_iconst(egraph, b) {
                    if d >= i32::MIN as i64 && d <= i32::MAX as i64 && a_ty == Type::I64 {
                        let lea4 = egraph.add(ENode {
                            op: Op::X86Lea4 {
                                scale: 1,
                                disp: d as i32,
                            },
                            children: smallvec![a, ClassId::NONE],
                        });
                        let canon = egraph.unionfind.find_immutable(class_id);
                        if egraph.unionfind.find_immutable(lea4) != canon {
                            egraph.merge(class_id, lea4);
                            changed = true;
                        }
                    }
                }
            }

            // Mul(a, 3) -> X86Lea3(a, a, 2)
            // Mul(a, 5) -> X86Lea3(a, a, 4)
            // Mul(a, 9) -> X86Lea3(a, a, 8)
            Op::Mul if snap.children.len() == 2 => {
                let a = snap.children[0];
                let b = snap.children[1];
                let (val_opt, base) = if let Some((v, _)) = find_iconst(egraph, b) {
                    (Some(v), a)
                } else if let Some((v, _)) = find_iconst(egraph, a) {
                    (Some(v), b)
                } else {
                    (None, a)
                };
                let base_ty = egraph
                    .class(egraph.unionfind.find_immutable(base))
                    .ty
                    .clone();
                if base_ty != Type::I64 {
                    continue;
                }
                if let Some(val) = val_opt {
                    let scale: Option<u8> = match val {
                        3 => Some(2),
                        5 => Some(4),
                        9 => Some(8),
                        _ => None,
                    };
                    if let Some(s) = scale {
                        let lea3 = egraph.add(ENode {
                            op: Op::X86Lea3 { scale: s },
                            children: smallvec![base, base],
                        });
                        let canon = egraph.unionfind.find_immutable(class_id);
                        if egraph.unionfind.find_immutable(lea3) != canon {
                            egraph.merge(class_id, lea3);
                            changed = true;
                        }
                    }
                }
            }

            _ => {}
        }
    }

    // Three-component LEA: Add(Add(a, Shl(b, Iconst(n))), Iconst(d))
    // -> X86Lea4(a, b, 2^n, d)
    changed |= apply_three_component_lea(egraph);

    changed
}

fn apply_three_component_lea(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        if snap.op != Op::Add || snap.children.len() != 2 {
            continue;
        }

        let outer_lhs = snap.children[0];
        let outer_rhs = snap.children[1];

        // Outer must have a constant displacement on one side
        let (disp_opt, inner_class) = if let Some((d, _)) = find_iconst(egraph, outer_rhs) {
            (Some(d), outer_lhs)
        } else if let Some((d, _)) = find_iconst(egraph, outer_lhs) {
            (Some(d), outer_rhs)
        } else {
            (None, outer_lhs)
        };

        let Some(disp) = disp_opt else { continue };
        if disp < i32::MIN as i64 || disp > i32::MAX as i64 {
            continue;
        }

        // Inner must be Add(a, Shl(b, Iconst(n)))
        let inner_canon = egraph.unionfind.find_immutable(inner_class);
        let inner_nodes: Vec<_> = egraph.class(inner_canon).nodes.clone();
        for inner_node in &inner_nodes {
            if inner_node.op != Op::Add || inner_node.children.len() != 2 {
                continue;
            }
            let [ia, ib] = [inner_node.children[0], inner_node.children[1]];

            // Check which child is Shl
            let (shl_class, base) = {
                let ia_canon = egraph.unionfind.find_immutable(ia);
                let ia_nodes: Vec<_> = egraph.class(ia_canon).nodes.clone();
                let has_shl_ia = ia_nodes
                    .iter()
                    .any(|n| n.op == Op::Shl && n.children.len() == 2);
                if has_shl_ia { (ia, ib) } else { (ib, ia) }
            };

            let shl_canon = egraph.unionfind.find_immutable(shl_class);
            let shl_nodes: Vec<_> = egraph.class(shl_canon).nodes.clone();
            for shl_node in &shl_nodes {
                if shl_node.op != Op::Shl || shl_node.children.len() != 2 {
                    continue;
                }
                let idx = shl_node.children[0];
                let shift = shl_node.children[1];
                let Some((n, _)) = find_iconst(egraph, shift) else {
                    continue;
                };
                if !matches!(n, 1 | 2 | 3) {
                    continue;
                }
                let scale = 1u8 << n;
                let base_ty = egraph
                    .class(egraph.unionfind.find_immutable(base))
                    .ty
                    .clone();
                let idx_ty = egraph
                    .class(egraph.unionfind.find_immutable(idx))
                    .ty
                    .clone();
                if base_ty != Type::I64 || idx_ty != Type::I64 {
                    continue;
                }
                let lea4 = egraph.add(ENode {
                    op: Op::X86Lea4 {
                        scale,
                        disp: disp as i32,
                    },
                    children: smallvec![base, idx],
                });
                let canon = egraph.unionfind.find_immutable(class_id);
                if egraph.unionfind.find_immutable(lea4) != canon {
                    egraph.merge(class_id, lea4);
                    changed = true;
                }
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;

    fn iconst(g: &mut EGraph, v: i64) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(v, Type::I64),
            children: smallvec![],
        })
    }

    fn var(g: &mut EGraph) -> ClassId {
        // Use a unique Iconst as a stand-in for an opaque variable
        static COUNTER: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1000);
        let v = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        g.add(ENode {
            op: Op::Iconst(v, Type::I64),
            children: smallvec![],
        })
    }

    // 4.17: Add(base, Iconst(16)) -> Addr(base, NONE, 1, 16)
    #[test]
    fn addr_base_plus_disp() {
        let mut g = EGraph::new();
        let base = var(&mut g);
        let disp = iconst(&mut g, 16);
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, disp],
        });
        apply_addr_mode_rules(&mut g);
        g.rebuild();

        let addr = g.add(ENode {
            op: Op::Addr { scale: 1, disp: 16 },
            children: smallvec![base, ClassId::NONE],
        });
        assert_eq!(g.find(add), g.find(addr));
    }

    // 4.17: Add(base, Shl(idx, Iconst(3))) -> Addr(base, idx, 8, 0)
    #[test]
    fn addr_base_plus_scaled_index() {
        let mut g = EGraph::new();
        let base = var(&mut g);
        let idx = var(&mut g);
        let three = iconst(&mut g, 3);
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![idx, three],
        });
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, shl],
        });
        apply_addr_mode_rules(&mut g);
        g.rebuild();

        let addr = g.add(ENode {
            op: Op::Addr { scale: 8, disp: 0 },
            children: smallvec![base, idx],
        });
        assert_eq!(g.find(add), g.find(addr));
    }

    // 4.17: invalid scale 6 rejected — no Addr{scale:6} created
    #[test]
    fn addr_invalid_scale_rejected() {
        let mut g = EGraph::new();
        let base = var(&mut g);
        let idx = var(&mut g);
        let six = iconst(&mut g, 6);
        let mul = g.add(ENode {
            op: Op::Mul,
            children: smallvec![idx, six],
        });
        let _add = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, mul],
        });
        apply_addr_mode_rules(&mut g);
        g.rebuild();

        // Verify no Addr{scale:6} was created
        for class in &g.classes {
            for node in &class.nodes {
                if let Op::Addr { scale: 6, .. } = node.op {
                    panic!("Addr with invalid scale=6 was created");
                }
            }
        }
    }

    // 4.17: Add(base, idx) -> Addr(base, idx, 1, 0)
    #[test]
    fn addr_base_plus_index() {
        let mut g = EGraph::new();
        let base = var(&mut g);
        let idx = var(&mut g);
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, idx],
        });
        apply_addr_mode_rules(&mut g);
        g.rebuild();

        let addr = g.add(ENode {
            op: Op::Addr { scale: 1, disp: 0 },
            children: smallvec![base, idx],
        });
        assert_eq!(g.find(add), g.find(addr));
    }

    // 4.17: Mul(a, 5) -> X86Lea3(a, a, 4)
    #[test]
    fn mul5_lea3() {
        let mut g = EGraph::new();
        let a = var(&mut g);
        let five = iconst(&mut g, 5);
        let mul = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, five],
        });
        apply_addr_mode_rules(&mut g);
        g.rebuild();

        let lea3 = g.add(ENode {
            op: Op::X86Lea3 { scale: 4 },
            children: smallvec![a, a],
        });
        assert_eq!(g.find(mul), g.find(lea3));
    }

    // 4.17: Three-component: Add(Add(a, Shl(b, 2)), Iconst(16)) -> X86Lea4(a, b, 4, 16)
    #[test]
    fn three_component_lea4() {
        let mut g = EGraph::new();
        let a = var(&mut g);
        let b = var(&mut g);
        let two = iconst(&mut g, 2);
        let sixteen = iconst(&mut g, 16);
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![b, two],
        });
        let inner_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, shl],
        });
        let outer_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![inner_add, sixteen],
        });
        // Run two passes so inner add gets its Addr/LEA forms first
        apply_addr_mode_rules(&mut g);
        g.rebuild();
        apply_addr_mode_rules(&mut g);
        g.rebuild();

        let lea4 = g.add(ENode {
            op: Op::X86Lea4 { scale: 4, disp: 16 },
            children: smallvec![a, b],
        });
        assert_eq!(g.find(outer_add), g.find(lea4));
    }
}
