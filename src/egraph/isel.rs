use smallvec::smallvec;

use crate::egraph::egraph::{EGraph, NodeSnap, snapshot_all};
use crate::egraph::enode::ENode;
use crate::ir::condcode::CondCode;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

pub fn apply_isel_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;
    changed |= apply_alu_isel(egraph, &snaps);
    changed |= apply_shift_isel(egraph, &snaps);
    changed |= apply_shift_imm_isel(egraph, &snaps);
    changed |= apply_select_isel(egraph, &snaps);
    changed |= apply_icmp_isel(egraph, &snaps);
    changed |= apply_fcmp_isel(egraph, &snaps);
    changed |= apply_sext_zext_trunc_isel(egraph, &snaps);
    changed |= apply_bitcast_isel(egraph, &snaps);
    changed |= apply_fp_isel(egraph, &snaps);
    changed |= apply_conv_isel(egraph, &snaps);
    changed |= apply_div_isel(egraph, &snaps);
    changed
}

/// SDiv(a,b) -> Proj0(X86Idiv(a,b))
/// SRem(a,b) -> Proj1(X86Idiv(a,b))
/// UDiv(a,b) -> Proj0(X86Div(a,b))
/// URem(a,b) -> Proj1(X86Div(a,b))
///
/// Egraph memoization ensures that SDiv and SRem on the same operands share
/// one X86Idiv node.
fn apply_div_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }

        let (x86_op, use_proj0) = match &snap.op {
            Op::SDiv => (Op::X86Idiv, true),
            Op::SRem => (Op::X86Idiv, false),
            Op::UDiv => (Op::X86Div, true),
            Op::URem => (Op::X86Div, false),
            _ => continue,
        };

        let a = snap.children[0];
        let b = snap.children[1];

        // Create (or reuse) X86Idiv/X86Div(a, b) — memo dedup handles sharing.
        let div_node = egraph.add(ENode {
            op: x86_op,
            children: smallvec![a, b],
        });

        let proj = egraph.add(ENode {
            op: if use_proj0 { Op::Proj0 } else { Op::Proj1 },
            children: smallvec![div_node],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let proj_canon = egraph.unionfind.find_immutable(proj);
        if canon != proj_canon {
            egraph.merge(class_id, proj);
            changed = true;
        }
    }
    changed
}

/// Map IR ALU binary ops to their x86 equivalents.
fn alu_x86_op(op: &Op) -> Option<Op> {
    match op {
        Op::Add => Some(Op::X86Add),
        Op::Sub => Some(Op::X86Sub),
        Op::And => Some(Op::X86And),
        Op::Or => Some(Op::X86Or),
        Op::Xor => Some(Op::X86Xor),
        Op::Mul => Some(Op::X86Imul3),
        _ => None,
    }
}

/// Add(a,b) -> Proj0(X86Add(a,b)), Sub(a,b) -> Proj0(X86Sub(a,b)), etc.
fn apply_alu_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
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
fn apply_shift_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
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

/// X86Shl(a, Iconst(n)) -> add X86ShlImm(n)(a) as an alternative in the same class.
/// Looks for Proj0(X86Shl(a, b)) where b has a constant value, and merges that
/// Proj0 class with Proj0(X86ShlImm(n)(a)). Same for Shr/Sar.
fn apply_shift_imm_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;

        // Look for Proj0 nodes whose child is X86Shl/X86Shr/X86Sar.
        if snap.op != Op::Proj0 || snap.children.len() != 1 {
            continue;
        }
        let shift_class = snap.children[0];
        let shift_canon = egraph.unionfind.find_immutable(shift_class);
        if shift_canon == ClassId::NONE {
            continue;
        }

        // Find an X86Shl/Shr/Sar node in the shift class with its operands.
        let shift_class_data = egraph.class(shift_canon);
        let shift_node = shift_class_data.nodes.iter().find(|n| {
            matches!(n.op, Op::X86Shl | Op::X86Shr | Op::X86Sar) && n.children.len() == 2
        });
        let Some(shift_node) = shift_node else {
            continue;
        };

        let mk_imm_op: fn(u8) -> Op = match &shift_node.op {
            Op::X86Shl => |n| Op::X86ShlImm(n),
            Op::X86Shr => |n| Op::X86ShrImm(n),
            Op::X86Sar => |n| Op::X86SarImm(n),
            _ => unreachable!(),
        };

        let a = shift_node.children[0];
        let b = shift_node.children[1];

        // Check if b is a constant that fits in shift count range 0..=63.
        let Some(val) = egraph.get_constant(b).map(|(v, _)| v) else {
            continue;
        };
        if !(0..=63).contains(&val) {
            continue;
        }
        let n = val as u8;

        // Create X86ShlImm(n)(a), then Proj0 of it.
        let imm_node = egraph.add(ENode {
            op: mk_imm_op(n),
            children: smallvec![a],
        });
        let proj0_imm = egraph.add(ENode {
            op: Op::Proj0,
            children: smallvec![imm_node],
        });

        // Merge the existing Proj0(X86Shl) class with Proj0(X86ShlImm).
        let canon = egraph.unionfind.find_immutable(class_id);
        let proj0_imm_canon = egraph.unionfind.find_immutable(proj0_imm);
        if canon != proj0_imm_canon {
            egraph.merge(class_id, proj0_imm);
            changed = true;
        }
    }
    changed
}

/// Icmp(cc, a, b) -> Proj1(X86Sub(a, b))
/// Multiple Icmps on same (a,b) share the same X86Sub.
///
/// Additionally, when b is an Iconst whose value fits in i32, add an
/// X86CmpI(imm) alternative to the flags class. The cost model makes that
/// cheaper than Proj1(X86Sub) (no register output, no iconst vreg), so
/// extraction picks it when the Sub's difference isn't otherwise needed.
fn apply_icmp_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
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

        // If RHS is an iconst fitting in i32, offer an X86CmpI alternative.
        // Extraction compares costs and picks X86CmpI when the Sub's
        // difference is unused (common: bare `if (x > n)` patterns).
        if let Some((v, _)) = egraph.get_constant(b)
            && let Ok(imm) = i32::try_from(v)
        {
            // The operand's integer type drives the compare width at lowering.
            // Grab it from the e-class `a` (all Icmp operands are integers).
            let a_canon = egraph.unionfind.find_immutable(a);
            let a_ty = egraph.class(a_canon).ty.clone();
            if a_ty.is_integer() {
                let x86cmpi = egraph.add(ENode {
                    op: Op::X86CmpI { imm, ty: a_ty },
                    children: smallvec![a],
                });
                let cmpi_canon = egraph.unionfind.find_immutable(x86cmpi);
                let canon2 = egraph.unionfind.find_immutable(class_id);
                if canon2 != cmpi_canon {
                    egraph.merge(class_id, x86cmpi);
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Fcmp(cc, a, b) -> X86Ucomisd(a, b) for F64, X86Ucomiss(a, b) for F32
/// The condition code is preserved in the Fcmp node for later extraction.
fn apply_fcmp_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let Op::Fcmp(_cc) = &snap.op else { continue };
        // OrdEq/UnordNe are composite CCs lowered directly (not via shared Ucomisd).
        // Skip isel to prevent merging with other Fcmp classes via Ucomisd hashcons.
        if matches!(_cc, CondCode::OrdEq | CondCode::UnordNe) {
            continue;
        }

        let a = snap.children[0];
        let b = snap.children[1];

        // Determine F32 vs F64 from the first operand type.
        let child_ty = infer_class_type(egraph, a);
        let is_f32 = matches!(child_ty, Some(Type::F32));

        let x86_op = if is_f32 {
            Op::X86Ucomiss
        } else {
            Op::X86Ucomisd
        };

        let ucomis = egraph.add(ENode {
            op: x86_op,
            children: smallvec![a, b],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let ucomis_canon = egraph.unionfind.find_immutable(ucomis);
        if canon != ucomis_canon {
            egraph.merge(class_id, ucomis);
            changed = true;
        }
    }
    changed
}

/// Select(flags, t, f) -> X86Cmov(cc, flags, t, f)
/// The cc is taken from the Icmp that produced the flags class.
fn apply_select_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.op != Op::Select || snap.children.len() != 3 {
            continue;
        }

        let flags = snap.children[0];
        let t = snap.children[1];
        let f = snap.children[2];

        // Find cc from the Icmp node in the flags class; fall back to Ne if absent.
        let cc = find_cc_in_class(egraph, flags).unwrap_or(CondCode::Ne);

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

/// Infer the result type of a class by inspecting its nodes.
///
/// Returns `Some(ty)` if any node in the class has a directly-determinable type
/// (constants, params, x86 machine ops, conversion ops). Returns `None` only if
/// no such node is found.
fn infer_class_type(egraph: &EGraph, class_id: ClassId) -> Option<Type> {
    let canon = egraph.unionfind.find_immutable(class_id);
    if canon == ClassId::NONE {
        return None;
    }
    let class = egraph.class(canon);
    // Use the type stored directly on the e-class (always available after `add`).
    Some(class.ty.clone())
}

/// Sext(ty)(a) -> X86Movsx{from, to}(a)
/// Zext(ty)(a) -> X86Movzx{from, to}(a)
/// Trunc(ty)(a) -> X86Trunc{from, to}(a)
fn apply_sext_zext_trunc_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 1 {
            continue;
        }

        let child = snap.children[0];
        let Some(from_ty) = infer_class_type(egraph, child) else {
            continue;
        };

        let machine_op = match &snap.op {
            Op::Sext(to) => Op::X86Movsx {
                from: from_ty,
                to: to.clone(),
            },
            Op::Zext(to) => Op::X86Movzx {
                from: from_ty,
                to: to.clone(),
            },
            Op::Trunc(to) => Op::X86Trunc {
                from: from_ty,
                to: to.clone(),
            },
            _ => continue,
        };

        let machine_node = egraph.add(ENode {
            op: machine_op,
            children: smallvec![child],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let machine_canon = egraph.unionfind.find_immutable(machine_node);
        if canon != machine_canon {
            egraph.merge(class_id, machine_node);
            changed = true;
        }
    }
    changed
}

/// Bitcast(to)(a) -> X86Bitcast{from, to}(a)
fn apply_bitcast_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 1 {
            continue;
        }
        let Op::Bitcast(to) = &snap.op else { continue };

        let child = snap.children[0];
        let Some(from_ty) = infer_class_type(egraph, child) else {
            continue;
        };

        let machine_node = egraph.add(ENode {
            op: Op::X86Bitcast {
                from: from_ty,
                to: to.clone(),
            },
            children: smallvec![child],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let machine_canon = egraph.unionfind.find_immutable(machine_node);
        if canon != machine_canon {
            egraph.merge(class_id, machine_node);
            changed = true;
        }
    }
    changed
}

/// Fadd/Fsub/Fmul/Fdiv/Fsqrt -> X86Addsd/X86Subsd/X86Mulsd/X86Divsd/X86Sqrtsd (F64)
///                             -> X86Addss/X86Subss/X86Mulss/X86Divss/X86Sqrtss (F32)
fn apply_fp_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;

        // Determine the operand type from the first child to choose sd vs ss.
        let child_ty = if !snap.children.is_empty() {
            infer_class_type(egraph, snap.children[0])
        } else {
            None
        };
        let is_f32 = matches!(child_ty, Some(Type::F32));

        let (machine_op, expected_children) = match &snap.op {
            Op::Fadd if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::X86Addss, 2)
                } else {
                    (Op::X86Addsd, 2)
                }
            }
            Op::Fsub if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::X86Subss, 2)
                } else {
                    (Op::X86Subsd, 2)
                }
            }
            Op::Fmul if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::X86Mulss, 2)
                } else {
                    (Op::X86Mulsd, 2)
                }
            }
            Op::Fdiv if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::X86Divss, 2)
                } else {
                    (Op::X86Divsd, 2)
                }
            }
            Op::Fsqrt if snap.children.len() == 1 => {
                if is_f32 {
                    (Op::X86Sqrtss, 1)
                } else {
                    (Op::X86Sqrtsd, 1)
                }
            }
            _ => continue,
        };

        let children: smallvec::SmallVec<[ClassId; 2]> =
            snap.children[..expected_children].iter().copied().collect();
        let machine_node = egraph.add(ENode {
            op: machine_op,
            children,
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let machine_canon = egraph.unionfind.find_immutable(machine_node);
        if canon != machine_canon {
            egraph.merge(class_id, machine_node);
            changed = true;
        }
    }
    changed
}

/// IntToFloat / FloatToInt / FloatExt / FloatTrunc -> x86 conversion ops
fn apply_conv_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 1 {
            continue;
        }

        let child = snap.children[0];

        let machine_op = match &snap.op {
            Op::IntToFloat(target) => match target {
                Type::F64 => Op::X86Cvtsi2sd,
                Type::F32 => Op::X86Cvtsi2ss,
                other => {
                    unreachable!("IntToFloat target must be F32 or F64, got {:?}", other);
                }
            },
            Op::FloatToInt(target) => {
                let child_ty = infer_class_type(egraph, child);
                let is_f32 = matches!(child_ty, Some(Type::F32));
                if is_f32 {
                    Op::X86Cvttss2si(target.clone())
                } else {
                    Op::X86Cvttsd2si(target.clone())
                }
            }
            Op::FloatExt => Op::X86Cvtss2sd,
            Op::FloatTrunc => Op::X86Cvtsd2ss,
            _ => continue,
        };

        let machine_node = egraph.add(ENode {
            op: machine_op,
            children: smallvec![child],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let machine_canon = egraph.unionfind.find_immutable(machine_node);
        if canon != machine_canon {
            egraph.merge(class_id, machine_node);
            changed = true;
        }
    }
    changed
}

/// Search the flags class for an Icmp node and extract its condition code.
pub(crate) fn find_cc_in_class(egraph: &EGraph, flags_class: ClassId) -> Option<CondCode> {
    let canon = egraph.unionfind.find_immutable(flags_class);
    if canon == ClassId::NONE {
        return None;
    }
    let class = egraph.class(canon);
    for node in &class.nodes {
        if let Op::Icmp(cc) = &node.op {
            return Some(*cc);
        }
        if let Op::Fcmp(cc) = &node.op {
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

    // Sext(I64) on an I32 value merges with X86Movsx{I32, I64}
    #[test]
    fn sext_i32_to_i64_isel() {
        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(42, Type::I32),
            children: smallvec![],
        });
        let sext = g.add(ENode {
            op: Op::Sext(Type::I64),
            children: smallvec![val],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let movsx = g.add(ENode {
            op: Op::X86Movsx {
                from: Type::I32,
                to: Type::I64,
            },
            children: smallvec![val],
        });
        assert_eq!(g.find(sext), g.find(movsx));
    }

    // Zext(I64) on an I8 value merges with X86Movzx{I8, I64}
    #[test]
    fn zext_i8_to_i64_isel() {
        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(1, Type::I8),
            children: smallvec![],
        });
        let zext = g.add(ENode {
            op: Op::Zext(Type::I64),
            children: smallvec![val],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let movzx = g.add(ENode {
            op: Op::X86Movzx {
                from: Type::I8,
                to: Type::I64,
            },
            children: smallvec![val],
        });
        assert_eq!(g.find(zext), g.find(movzx));
    }

    // Trunc(I32) on an I64 value merges with X86Trunc{I64, I32}
    #[test]
    fn trunc_i64_to_i32_isel() {
        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Iconst(0xFF_FFFF_FFFFi64, Type::I64),
            children: smallvec![],
        });
        let trunc = g.add(ENode {
            op: Op::Trunc(Type::I32),
            children: smallvec![val],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let x86trunc = g.add(ENode {
            op: Op::X86Trunc {
                from: Type::I64,
                to: Type::I32,
            },
            children: smallvec![val],
        });
        assert_eq!(g.find(trunc), g.find(x86trunc));
    }
}
