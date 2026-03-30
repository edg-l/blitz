use smallvec::smallvec;

use crate::egraph::egraph::EGraph;
use crate::egraph::enode::ENode;
use crate::ir::condcode::CondCode;
use crate::ir::op::{ClassId, Op};

pub fn apply_isel_rules(egraph: &mut EGraph) -> bool {
    let mut changed = false;
    changed |= apply_alu_isel(egraph);
    changed |= apply_shift_isel(egraph);
    changed |= apply_icmp_isel(egraph);
    changed |= apply_select_isel(egraph);
    changed
}

/// Snapshot of one node for mutation-safe iteration.
struct Snap {
    class_id: ClassId,
    op: Op,
    children: smallvec::SmallVec<[ClassId; 2]>,
}

fn snapshot(egraph: &EGraph) -> Vec<Snap> {
    let mut snaps = Vec::new();
    for i in 0..egraph.classes.len() as u32 {
        let id = ClassId(i);
        if egraph.unionfind.find_immutable(id) != id {
            continue;
        }
        let class = egraph.class(id);
        for node in &class.nodes {
            snaps.push(Snap {
                class_id: id,
                op: node.op.clone(),
                children: node.children.clone(),
            });
        }
    }
    snaps
}

/// Map IR ALU binary ops to their x86 equivalents.
fn alu_x86_op(op: &Op) -> Option<Op> {
    match op {
        Op::Add => Some(Op::X86Add),
        Op::Sub => Some(Op::X86Sub),
        Op::And => Some(Op::X86And),
        Op::Or => Some(Op::X86Or),
        Op::Xor => Some(Op::X86Xor),
        _ => None,
    }
}

/// Add(a,b) -> Proj0(X86Add(a,b)), Sub(a,b) -> Proj0(X86Sub(a,b)), etc.
fn apply_alu_isel(egraph: &mut EGraph) -> bool {
    let snaps = snapshot(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let Some(x86_op) = alu_x86_op(&snap.op) else {
            continue;
        };

        let a = snap.children[0];
        let b = snap.children[1];

        // Create X86Op(a, b)
        let x86_node = egraph.add(ENode {
            op: x86_op,
            children: smallvec![a, b],
        });

        // Proj0 extracts the value result
        let proj0 = egraph.add(ENode {
            op: Op::Proj0,
            children: smallvec![x86_node],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let proj0_canon = egraph.unionfind.find_immutable(proj0);
        if canon != proj0_canon {
            egraph.merge(class_id, proj0);
            changed = true;
        }
    }
    changed
}

/// Shl/Sar/Shr -> X86Shl/X86Sar/X86Shr (as Proj0)
fn apply_shift_isel(egraph: &mut EGraph) -> bool {
    let snaps = snapshot(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let x86_op = match &snap.op {
            Op::Shl => Op::X86Shl,
            Op::Sar => Op::X86Sar,
            Op::Shr => Op::X86Shr,
            _ => continue,
        };

        let a = snap.children[0];
        let b = snap.children[1];

        let x86_node = egraph.add(ENode {
            op: x86_op,
            children: smallvec![a, b],
        });
        let proj0 = egraph.add(ENode {
            op: Op::Proj0,
            children: smallvec![x86_node],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let proj0_canon = egraph.unionfind.find_immutable(proj0);
        if canon != proj0_canon {
            egraph.merge(class_id, proj0);
            changed = true;
        }
    }
    changed
}

/// Icmp(cc, a, b) -> Proj1(X86Sub(a, b))
/// Multiple Icmps on same (a,b) share the same X86Sub.
fn apply_icmp_isel(egraph: &mut EGraph) -> bool {
    let snaps = snapshot(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let Op::Icmp(_cc) = &snap.op else { continue };

        let a = snap.children[0];
        let b = snap.children[1];

        // Create (or reuse) X86Sub(a, b) — memo dedup handles reuse
        let x86sub = egraph.add(ENode {
            op: Op::X86Sub,
            children: smallvec![a, b],
        });

        // Proj1 is the FLAGS output
        let proj1 = egraph.add(ENode {
            op: Op::Proj1,
            children: smallvec![x86sub],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let proj1_canon = egraph.unionfind.find_immutable(proj1);
        if canon != proj1_canon {
            egraph.merge(class_id, proj1);
            changed = true;
        }
    }
    changed
}

/// Select(flags, t, f) -> X86Cmov(cc, flags, t, f)
/// The cc is taken from the Icmp that produced the flags class.
fn apply_select_isel(egraph: &mut EGraph) -> bool {
    let snaps = snapshot(egraph);
    let mut changed = false;

    for snap in &snaps {
        let class_id = snap.class_id;
        if snap.op != Op::Select || snap.children.len() != 3 {
            continue;
        }

        let flags = snap.children[0];
        let t = snap.children[1];
        let f = snap.children[2];

        // Find cc from the Icmp (or X86Cmov) node in the flags class
        let cc = find_cc_in_class(egraph, flags);
        let Some(cc) = cc else { continue };

        let cmov = egraph.add(ENode {
            op: Op::X86Cmov(cc),
            children: smallvec![flags, t, f],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let cmov_canon = egraph.unionfind.find_immutable(cmov);
        if canon != cmov_canon {
            egraph.merge(class_id, cmov);
            changed = true;
        }
    }
    changed
}

/// Search the flags class for an Icmp node and extract its condition code.
fn find_cc_in_class(egraph: &EGraph, flags_class: ClassId) -> Option<CondCode> {
    let canon = egraph.unionfind.find_immutable(flags_class);
    if canon == ClassId::NONE {
        return None;
    }
    let class = egraph.class(canon);
    for node in &class.nodes {
        if let Op::Icmp(cc) = &node.op {
            return Some(*cc);
        }
        // Also look through Proj1 -> X86Sub nodes: the cc comes from the original Icmp
        // which is in the same e-class after merging, so the Icmp node is found above.
    }
    None
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;
    use crate::ir::types::Type;

    fn iconst(g: &mut EGraph, v: i64) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(v, Type::I64),
            children: smallvec![],
        })
    }

    // 4.14: Add(a,b) -> merges with Proj0(X86Add(a,b))
    #[test]
    fn add_isel_to_x86add() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1);
        let b = iconst(&mut g, 2);
        let ir_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, b],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let x86add = g.add(ENode {
            op: Op::X86Add,
            children: smallvec![a, b],
        });
        let proj0 = g.add(ENode {
            op: Op::Proj0,
            children: smallvec![x86add],
        });
        assert_eq!(g.find(ir_add), g.find(proj0));
    }

    // 4.14: Sub(a,b) and Icmp(Slt,a,b) share X86Sub
    #[test]
    fn sub_and_icmp_share_x86sub() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 10);
        let b = iconst(&mut g, 5);
        let ir_sub = g.add(ENode {
            op: Op::Sub,
            children: smallvec![a, b],
        });
        let icmp = g.add(ENode {
            op: Op::Icmp(CondCode::Slt),
            children: smallvec![a, b],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        // Both should reference the same X86Sub
        let x86sub = g.add(ENode {
            op: Op::X86Sub,
            children: smallvec![a, b],
        });
        let proj0 = g.add(ENode {
            op: Op::Proj0,
            children: smallvec![x86sub],
        });
        let proj1 = g.add(ENode {
            op: Op::Proj1,
            children: smallvec![x86sub],
        });
        assert_eq!(g.find(ir_sub), g.find(proj0));
        assert_eq!(g.find(icmp), g.find(proj1));
    }

    // 4.14: Two Icmp with different cc on same operands share one X86Sub
    #[test]
    fn two_icmps_share_x86sub() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 10);
        let b = iconst(&mut g, 5);
        let icmp_slt = g.add(ENode {
            op: Op::Icmp(CondCode::Slt),
            children: smallvec![a, b],
        });
        let icmp_ult = g.add(ENode {
            op: Op::Icmp(CondCode::Ult),
            children: smallvec![a, b],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let x86sub = g.add(ENode {
            op: Op::X86Sub,
            children: smallvec![a, b],
        });
        let proj1 = g.add(ENode {
            op: Op::Proj1,
            children: smallvec![x86sub],
        });
        // Both icmp classes merge with the same Proj1(X86Sub)
        assert_eq!(g.find(icmp_slt), g.find(proj1));
        assert_eq!(g.find(icmp_ult), g.find(proj1));
    }

    // 4.14: Select(flags, t, f) -> X86Cmov
    #[test]
    fn select_isel_to_cmov() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 10);
        let b = iconst(&mut g, 5);
        let flags = g.add(ENode {
            op: Op::Icmp(CondCode::Eq),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1);
        let f = iconst(&mut g, 0);
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![flags, t, f],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let cmov = g.add(ENode {
            op: Op::X86Cmov(CondCode::Eq),
            children: smallvec![flags, t, f],
        });
        assert_eq!(g.find(sel), g.find(cmov));
    }
}
