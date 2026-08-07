use smallvec::smallvec;

use crate::egraph::egraph::{EGraph, NodeSnap, snapshot_all};
use crate::egraph::enode::ENode;
use crate::ir::condcode::CondCode;
use crate::ir::op::{ClassId, MachOp, Op, PureOp};
use crate::ir::types::Type;

pub fn apply_isel_rules(egraph: &mut EGraph) -> bool {
    let snaps = snapshot_all(egraph);
    let mut changed = false;
    changed |= apply_alu_isel(egraph, &snaps);
    changed |= apply_shift_isel(egraph, &snaps);
    changed |= apply_shift_imm_isel(egraph, &snaps);
    changed |= apply_funnel_shift_isel(egraph, &snaps);
    changed |= apply_bit_isel(egraph, &snaps);
    changed |= apply_alu_imm_isel(egraph, &snaps);
    changed |= apply_select_isel(egraph, &snaps);
    changed |= apply_carry_mask_isel(egraph, &snaps);
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

        let a = snap.children[0];
        let b = snap.children[1];

        // The operand type, not the class type: the class of an SRem is the
        // remainder's type, which is the same integer type, but taking it from the
        // operand is what the op documents and what lowering needs.
        let operand_ty = egraph.class(egraph.unionfind.find_immutable(a)).ty.clone();
        if !operand_ty.is_integer() {
            continue;
        }
        let (x86_op, use_proj0) = match &snap.op {
            Op::Pure(PureOp::SDiv) => (Op::Mach(MachOp::X86Idiv(operand_ty)), true),
            Op::Pure(PureOp::SRem) => (Op::Mach(MachOp::X86Idiv(operand_ty)), false),
            Op::Pure(PureOp::UDiv) => (Op::Mach(MachOp::X86Div(operand_ty)), true),
            Op::Pure(PureOp::URem) => (Op::Mach(MachOp::X86Div(operand_ty)), false),
            _ => continue,
        };

        // Create (or reuse) X86Idiv/X86Div(a, b) — memo dedup handles sharing.
        let div_node = egraph.add(ENode {
            op: x86_op,
            children: smallvec![a, b],
        });

        let proj = egraph.add(ENode {
            op: if use_proj0 {
                Op::Pure(PureOp::Proj0)
            } else {
                Op::Pure(PureOp::Proj1)
            },
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
        Op::Pure(PureOp::Add) => Some(Op::Mach(MachOp::X86Add)),
        Op::Pure(PureOp::Sub) => Some(Op::Mach(MachOp::X86Sub)),
        Op::Pure(PureOp::And) => Some(Op::Mach(MachOp::X86And)),
        Op::Pure(PureOp::Or) => Some(Op::Mach(MachOp::X86Or)),
        Op::Pure(PureOp::Xor) => Some(Op::Mach(MachOp::X86Xor)),
        Op::Pure(PureOp::Mul) => Some(Op::Mach(MachOp::X86Imul3)),
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
            op: Op::Pure(PureOp::Proj0),
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
            Op::Pure(PureOp::Shl) => Op::Mach(MachOp::X86Shl),
            Op::Pure(PureOp::Sar) => Op::Mach(MachOp::X86Sar),
            Op::Pure(PureOp::Shr) => Op::Mach(MachOp::X86Shr),
            _ => continue,
        };

        let a = snap.children[0];
        let b = snap.children[1];

        let x86_node = egraph.add(ENode {
            op: x86_op,
            children: smallvec![a, b],
        });
        let proj0 = egraph.add(ENode {
            op: Op::Pure(PureOp::Proj0),
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

/// `X86Add(a, Iconst(n))` -> add `X86AddI(n)(a)` as an alternative in the same
/// class, and the same for Sub/And/Or/Xor.
///
/// Both forms are one instruction, so what the immediate form saves is the
/// other operand: an `Iconst` selected as a register operand costs a register
/// and the `mov` that fills it, for the whole of its live range. The cost model
/// prices only the node, so it carries a small discount to break the tie; the
/// real win shows up as one fewer value live.
///
/// `Sub` takes the constant on the right only. `c - x` has no immediate form --
/// the immediate is the subtrahend -- and the algebraic rules do not turn one
/// into the other, so matching either side here would emit `sub x, c` for
/// `c - x`.
fn apply_alu_imm_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    /// A constructor for the immediate form of an ALU op, and whether its
    /// operands commute.
    type ImmForm = (fn(i32) -> Op, bool);

    fn imm_form(op: &Op) -> Option<ImmForm> {
        match op {
            Op::Mach(MachOp::X86Add) => {
                Some(((|n| Op::Mach(MachOp::X86AddI(n))) as fn(i32) -> Op, true))
            }
            Op::Mach(MachOp::X86Sub) => {
                Some(((|n| Op::Mach(MachOp::X86SubI(n))) as fn(i32) -> Op, false))
            }
            Op::Mach(MachOp::X86And) => {
                Some(((|n| Op::Mach(MachOp::X86AndI(n))) as fn(i32) -> Op, true))
            }
            Op::Mach(MachOp::X86Or) => {
                Some(((|n| Op::Mach(MachOp::X86OrI(n))) as fn(i32) -> Op, true))
            }
            Op::Mach(MachOp::X86Xor) => {
                Some(((|n| Op::Mach(MachOp::X86XorI(n))) as fn(i32) -> Op, true))
            }
            _ => None,
        }
    }

    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;

        // The pair itself is not the value; `Proj0` of it is, and that is the
        // class an alternative has to join.
        if snap.op != Op::Pure(PureOp::Proj0) || snap.children.len() != 1 {
            continue;
        }
        let alu_canon = egraph.unionfind.find_immutable(snap.children[0]);
        if alu_canon == ClassId::NONE {
            continue;
        }

        // Take the operands and the form out of the class before touching the
        // e-graph: `egraph.add` below invalidates this borrow.
        let found = egraph.class(alu_canon).nodes.iter().find_map(|n| {
            let (mk, commutes) = imm_form(&n.op)?;
            if n.children.len() != 2 {
                return None;
            }
            Some((mk, commutes, n.children[0], n.children[1]))
        });
        let Some((mk, commutes, a, b)) = found else {
            continue;
        };

        // The register operand and the constant, whichever side the constant is
        // on. A node with two constants is left alone: folding it is the
        // algebraic rules' job, and selecting one operand as an immediate would
        // keep the other in a register for nothing.
        let rhs_const = egraph.get_constant(b).map(|(v, _)| v);
        let lhs_const = commutes
            .then(|| egraph.get_constant(a).map(|(v, _)| v))
            .flatten();
        let (reg_operand, konst) = match (lhs_const, rhs_const) {
            (_, Some(v)) => (a, v),
            (Some(v), None) => (b, v),
            (None, None) => continue,
        };
        let Ok(imm) = i32::try_from(konst) else {
            continue;
        };

        let imm_node = egraph.add(ENode {
            op: mk(imm),
            children: smallvec![reg_operand],
        });
        let proj0_imm = egraph.add(ENode {
            op: Op::Pure(PureOp::Proj0),
            children: smallvec![imm_node],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let proj0_imm_canon = egraph.unionfind.find_immutable(proj0_imm);
        if canon != proj0_imm_canon {
            egraph.merge(class_id, proj0_imm);
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
        if snap.op != Op::Pure(PureOp::Proj0) || snap.children.len() != 1 {
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
            matches!(
                n.op,
                Op::Mach(MachOp::X86Shl) | Op::Mach(MachOp::X86Shr) | Op::Mach(MachOp::X86Sar)
            ) && n.children.len() == 2
        });
        let Some(shift_node) = shift_node else {
            continue;
        };

        let mk_imm_op: fn(u8) -> Op = match &shift_node.op {
            Op::Mach(MachOp::X86Shl) => |n| Op::Mach(MachOp::X86ShlImm(n)),
            Op::Mach(MachOp::X86Shr) => |n| Op::Mach(MachOp::X86ShrImm(n)),
            Op::Mach(MachOp::X86Sar) => |n| Op::Mach(MachOp::X86SarImm(n)),
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
            op: Op::Pure(PureOp::Proj0),
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

/// `Or(Shl(x, k), Shr(y, w - k))` on `w`-bit operands -> one funnel shift:
/// `Proj0(X86RolImm(k)(x))` where `x` and `y` are the same value, and
/// `Proj0(X86ShldImm(k)(x, y))` where they differ.
///
/// Either is one instruction where the shift pair is three, and the rotate reads
/// `x` once rather than twice, so the value it rotates does not have to stay live
/// across a second shift. `Shr` and not `Sar`: an arithmetic shift feeds the sign
/// bit into the high end, which is what neither form puts there.
///
/// The two amounts must sum to the width of the shifted values' own type, so a
/// `w`-bit funnel shift expressed on a wider type is not matched -- there the high
/// bits of the shift-left result survive and the single instruction would drop
/// them.
fn apply_funnel_shift_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    /// The two shifted values and the left shift amount, read off one operand
    /// class pair of an `Or` taken in the order (shifted left, shifted right).
    fn funnel_parts(
        egraph: &EGraph,
        shl_class: ClassId,
        shr_class: ClassId,
        width: u32,
    ) -> Option<(ClassId, ClassId, u8)> {
        let shift = |class: ClassId, want: PureOp| -> Option<(ClassId, i64)> {
            let canon = egraph.unionfind.find_immutable(class);
            if canon == ClassId::NONE {
                return None;
            }
            egraph.class(canon).nodes.iter().find_map(|n| {
                (n.op == Op::Pure(want.clone()) && n.children.len() == 2)
                    .then(|| {
                        egraph
                            .get_constant(n.children[1])
                            .map(|(v, _)| (n.children[0], v))
                    })
                    .flatten()
            })
        };

        let (x, k) = shift(shl_class, PureOp::Shl)?;
        let (y, j) = shift(shr_class, PureOp::Shr)?;
        let x_canon = egraph.unionfind.find_immutable(x);
        let y_canon = egraph.unionfind.find_immutable(y);
        if x_canon == ClassId::NONE || y_canon == ClassId::NONE {
            return None;
        }
        if k <= 0 || j <= 0 || k + j != i64::from(width) {
            return None;
        }
        // The funnel shift is over the whole of its own type, so both values it
        // reads must have that type and not a wider one truncated into it.
        let full_width = |class: ClassId| {
            infer_class_type(egraph, class)
                .is_some_and(|ty| ty.is_integer() && ty.bit_width() == width)
        };
        (full_width(x_canon) && full_width(y_canon)).then_some((x_canon, y_canon, k as u8))
    }

    let mut changed = false;

    for snap in snaps {
        if snap.op != Op::Pure(PureOp::Or) || snap.children.len() != 2 {
            continue;
        }
        let Some(ty) = infer_class_type(egraph, snap.class_id) else {
            continue;
        };
        if !ty.is_integer() {
            continue;
        }
        let width = ty.bit_width();
        let (a, b) = (snap.children[0], snap.children[1]);
        let Some((x, y, k)) =
            funnel_parts(egraph, a, b, width).or_else(|| funnel_parts(egraph, b, a, width))
        else {
            continue;
        };

        // `shld` has no byte form, so an 8-bit funnel shift of two values stays
        // as the shift pair; an 8-bit rotate is still a rotate.
        let node = if x == y {
            ENode {
                op: Op::Mach(MachOp::X86RolImm(k)),
                children: smallvec![x],
            }
        } else if width > 8 {
            ENode {
                op: Op::Mach(MachOp::X86ShldImm(k)),
                children: smallvec![x, y],
            }
        } else {
            continue;
        };

        let funnel = egraph.add(node);
        let proj0 = egraph.add(ENode {
            op: Op::Pure(PureOp::Proj0),
            children: smallvec![funnel],
        });

        let canon = egraph.unionfind.find_immutable(snap.class_id);
        let proj0_canon = egraph.unionfind.find_immutable(proj0);
        if canon != proj0_canon {
            egraph.merge(snap.class_id, proj0);
            changed = true;
        }
    }
    changed
}

/// A one-bit mask built at run time collapses into the single-bit instruction
/// that reads its index directly:
///
/// - `Or(x, Shl(1, n))`             -> `Proj0(X86Bts(x, n))`
/// - `Xor(x, Shl(1, n))`            -> `Proj0(X86Btc(x, n))`
/// - `And(x, Xor(Shl(1, n), -1))`   -> `Proj0(X86Btr(x, n))`
///
/// The mask form costs three instructions: the `1` into a register, a shift that
/// must route `n` through CL, and the ALU op itself; `btr` pays a fourth for the
/// complement. The bit instructions take the index from any register, so the
/// whole sequence becomes one instruction and CL stops being contended.
///
/// A constant `n` folds the shift away, leaving the mask itself, which the
/// same three take by immediate:
///
/// - `Or(x, 1 << k)`     -> `Proj0(X86BtsI(k)(x))`
/// - `Xor(x, 1 << k)`    -> `Proj0(X86BtcI(k)(x))`
/// - `And(x, !(1 << k))` -> `Proj0(X86BtrI(k)(x))`
///
/// only from bit 7 up. Below that the mask is an `imm8` and the immediate-form
/// ALU already has it in three bytes where `bts` needs four; at 7 and above the
/// register form has to materialize the mask, five bytes for a `mov r, imm32`
/// and ten for a 64-bit one.
///
/// The bit index is taken modulo the operand width by the hardware. A `1 << n`
/// whose `n` reaches the width is undefined in C, so the two agree wherever the
/// source has a meaning. There is no byte form, so 8-bit masks stay as they are.
fn apply_bit_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    /// The bit index of a `Shl(1, n)` in `class`, with `n` not a constant.
    fn bit_index(egraph: &EGraph, class: ClassId) -> Option<ClassId> {
        let canon = egraph.unionfind.find_immutable(class);
        if canon == ClassId::NONE {
            return None;
        }
        egraph.class(canon).nodes.iter().find_map(|node| {
            if node.op != Op::Pure(PureOp::Shl) || node.children.len() != 2 {
                return None;
            }
            let (one, n) = (node.children[0], node.children[1]);
            if egraph.get_constant(one)?.0 != 1 || egraph.get_constant(n).is_some() {
                return None;
            }
            let n = egraph.unionfind.find_immutable(n);
            (n != ClassId::NONE).then_some(n)
        })
    }

    /// The bit a constant one-bit mask in `class` sets, on a `width`-bit
    /// operand, taking the mask's complement first for the `And` form.
    fn const_bit_index(
        egraph: &EGraph,
        class: ClassId,
        width: u32,
        complement: bool,
    ) -> Option<u8> {
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let v = egraph.get_constant(class)?.0 as u64 & mask;
        let v = if complement { !v & mask } else { v };
        (v.count_ones() == 1).then(|| v.trailing_zeros() as u8)
    }

    /// The bit index of an all-ones complement of a one-bit mask, `Xor(Shl(1, n), -1)`.
    fn complemented_bit_index(egraph: &EGraph, class: ClassId) -> Option<ClassId> {
        let canon = egraph.unionfind.find_immutable(class);
        if canon == ClassId::NONE {
            return None;
        }
        egraph.class(canon).nodes.iter().find_map(|node| {
            if node.op != Op::Pure(PureOp::Xor) || node.children.len() != 2 {
                return None;
            }
            let (a, b) = (node.children[0], node.children[1]);
            let ones = |c: ClassId| egraph.get_constant(c).is_some_and(|(v, _)| v == -1);
            if ones(b) {
                bit_index(egraph, a)
            } else if ones(a) {
                bit_index(egraph, b)
            } else {
                None
            }
        })
    }

    let mut changed = false;

    for snap in snaps {
        let mach = match snap.op {
            Op::Pure(PureOp::Or) => MachOp::X86Bts,
            Op::Pure(PureOp::Xor) => MachOp::X86Btc,
            Op::Pure(PureOp::And) => MachOp::X86Btr,
            _ => continue,
        };
        if snap.children.len() != 2 {
            continue;
        }
        // `bt` has no byte form, and the value operated on must be the whole of
        // its own type: a mask on a wider type truncated into this one would set
        // a bit the single instruction cannot reach.
        let Some(ty) = infer_class_type(egraph, snap.class_id) else {
            continue;
        };
        if !ty.is_integer() || ty.bit_width() < 16 {
            continue;
        }
        let width = ty.bit_width();
        let complement = mach == MachOp::X86Btr;
        let index = if complement {
            complemented_bit_index
        } else {
            bit_index
        };

        let (a, b) = (snap.children[0], snap.children[1]);
        // The value operated on is read at the instruction's own width, and so
        // is a register index: a narrower one would be read together with
        // whatever sits above it.
        let full_width = |c: ClassId| {
            let c = egraph.unionfind.find_immutable(c);
            c != ClassId::NONE
                && infer_class_type(egraph, c)
                    .is_some_and(|t| t.is_integer() && t.bit_width() == width)
        };

        let variable = index(egraph, b)
            .map(|n| (a, n))
            .or_else(|| index(egraph, a).map(|n| (b, n)))
            .filter(|&(x, n)| full_width(x) && full_width(n));
        // A one-bit constant mask below bit 7 is left alone: the immediate-form
        // ALU reaches it in three bytes where `bts` needs four. At 7 and above
        // the mask no longer fits an `imm8` and the register form has to
        // materialize it.
        let constant = const_bit_index(egraph, b, width, complement)
            .map(|k| (a, k))
            .or_else(|| const_bit_index(egraph, a, width, complement).map(|k| (b, k)))
            .filter(|&(x, k)| k >= 7 && full_width(x) && egraph.get_constant(x).is_none());

        let (op, children) = if let Some((x, n)) = variable {
            let (x, n) = (
                egraph.unionfind.find_immutable(x),
                egraph.unionfind.find_immutable(n),
            );
            (Op::Mach(mach), smallvec![x, n])
        } else if let Some((x, k)) = constant {
            let mach = match mach {
                MachOp::X86Bts => MachOp::X86BtsI(k),
                MachOp::X86Btc => MachOp::X86BtcI(k),
                _ => MachOp::X86BtrI(k),
            };
            (
                Op::Mach(mach),
                smallvec![egraph.unionfind.find_immutable(x)],
            )
        } else {
            continue;
        };

        let bit = egraph.add(ENode { op, children });
        let proj0 = egraph.add(ENode {
            op: Op::Pure(PureOp::Proj0),
            children: smallvec![bit],
        });

        let canon = egraph.unionfind.find_immutable(snap.class_id);
        let proj0_canon = egraph.unionfind.find_immutable(proj0);
        if canon != proj0_canon {
            egraph.merge(snap.class_id, proj0);
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
        let Op::Pure(PureOp::Icmp(_cc)) = &snap.op else {
            continue;
        };

        let a = snap.children[0];
        let b = snap.children[1];

        // Create (or reuse) X86Sub(a, b) — memo dedup handles reuse
        let x86sub = egraph.add(ENode {
            op: Op::Mach(MachOp::X86Sub),
            children: smallvec![a, b],
        });

        // Proj1 is the FLAGS output
        let proj1 = egraph.add(ENode {
            op: Op::Pure(PureOp::Proj1),
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
                    op: Op::Mach(MachOp::X86CmpI { imm, ty: a_ty }),
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

/// `Fcmp(cc, a, b)` -> `X86Ucomisd(a, b)` for F64, `X86Ucomiss(a, b)` for F32.
///
/// The condition code stays on the `Fcmp` node for later extraction, except for
/// the composite codes, which carry it on the machine node instead -- see
/// [`MachOp::X86UcomisdCc`].
fn apply_fcmp_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.children.len() != 2 {
            continue;
        }
        let Op::Pure(PureOp::Fcmp(cc)) = &snap.op else {
            continue;
        };
        let cc = *cc;

        let a = snap.children[0];
        let b = snap.children[1];

        // Determine F32 vs F64 from the first operand type.
        let child_ty = infer_class_type(egraph, a);
        let is_f32 = matches!(child_ty, Some(Type::F32));

        // A composite condition code needs two flag tests, so it cannot share the
        // plain `X86Ucomisd` node with the one-test codes: hashconsing would give
        // every comparison of the same pair one class and the code would be lost.
        // Carrying it on the node keeps them distinct, which is what lets these
        // reach `lower_op` like any other machine op instead of being lowered by
        // a separate path that bypasses isel.
        let composite = matches!(cc, CondCode::OrdEq | CondCode::UnordNe);
        let x86_op = match (composite, is_f32) {
            (true, true) => Op::Mach(MachOp::X86UcomissCc(cc)),
            (true, false) => Op::Mach(MachOp::X86UcomisdCc(cc)),
            (false, true) => Op::Mach(MachOp::X86Ucomiss),
            (false, false) => Op::Mach(MachOp::X86Ucomisd),
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
        if snap.op != Op::Pure(PureOp::Select) || snap.children.len() != 3 {
            continue;
        }

        let flags = snap.children[0];
        let t = snap.children[1];
        let f = snap.children[2];

        // Find cc from the Icmp node in the flags class; fall back to Ne if absent.
        let cc = find_cc_in_class(egraph, flags).unwrap_or(CondCode::Ne);

        let cmov = egraph.add(ENode {
            op: Op::Mach(MachOp::X86Cmov(cc)),
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

/// `Sub(0, Select(flags, 1, 0))` -> `X86SbbSelf(flags)` when the flags come
/// from an unsigned-below compare.
///
/// `sbb r, r` subtracts a register from itself with borrow, so it leaves `-CF`:
/// all ones when the carry flag is set, zero when it is clear. After
/// `cmp a, b` the carry flag *is* `a < b` unsigned, so the whole 0/-1 mask is
/// two bytes and one instruction where the select form is five -- the two
/// constants into registers, the `cmov`, and the subtract -- and it holds no
/// register but its own result across the compare.
///
/// Only `Ult`. The other conditions are not the carry flag: `Ugt` and `Ule`
/// would need the compare's operands swapped, which is a different compare,
/// and the signed ones are not carry at all.
///
/// A widening between the select and the subtract is transparent. C gives the
/// comparison type `int`, so a mask wider than that arrives as
/// `Sub(0, Sext(Select(..)))`, and extending a value that is already 0 or 1
/// leaves it 0 or 1 whichever way the extension fills.
fn apply_carry_mask_isel(egraph: &mut EGraph, snaps: &[NodeSnap]) -> bool {
    /// The flags class of a `Select(flags, 1, 0)` in `class`, looking through
    /// up to `exts` extensions.
    fn zero_one_select_flags(egraph: &EGraph, class: ClassId, exts: u32) -> Option<ClassId> {
        let canon = egraph.unionfind.find_immutable(class);
        if canon == ClassId::NONE {
            return None;
        }
        let mut extended = None;
        for node in &egraph.class(canon).nodes {
            match &node.op {
                Op::Pure(PureOp::Select)
                    if node.children.len() == 3
                        && egraph.get_constant(node.children[1]).map(|(v, _)| v) == Some(1)
                        && egraph.get_constant(node.children[2]).map(|(v, _)| v) == Some(0) =>
                {
                    return Some(node.children[0]);
                }
                Op::Pure(PureOp::Sext(_) | PureOp::Zext(_))
                    if exts > 0 && node.children.len() == 1 =>
                {
                    extended.get_or_insert(node.children[0]);
                }
                _ => {}
            }
        }
        extended.and_then(|inner| zero_one_select_flags(egraph, inner, exts - 1))
    }

    let mut changed = false;

    for snap in snaps {
        let class_id = snap.class_id;
        if snap.op != Op::Pure(PureOp::Sub) || snap.children.len() != 2 {
            continue;
        }
        if egraph.get_constant(snap.children[0]).map(|(v, _)| v) != Some(0) {
            continue;
        }

        // The flags class, taken before `egraph.add` invalidates the borrow.
        let Some(flags) = zero_one_select_flags(egraph, snap.children[1], 1) else {
            continue;
        };
        if find_cc_in_class(egraph, flags) != Some(CondCode::Ult) {
            continue;
        }

        let ty = egraph
            .class(egraph.unionfind.find_immutable(class_id))
            .ty
            .clone();
        if !ty.is_integer() {
            continue;
        }

        let sbb = egraph.add(ENode {
            op: Op::Mach(MachOp::X86SbbSelf(ty)),
            children: smallvec![flags],
        });

        let canon = egraph.unionfind.find_immutable(class_id);
        let sbb_canon = egraph.unionfind.find_immutable(sbb);
        if canon != sbb_canon {
            egraph.merge(class_id, sbb);
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
            Op::Pure(PureOp::Sext(to)) => Op::Mach(MachOp::X86Movsx {
                from: from_ty,
                to: to.clone(),
            }),
            Op::Pure(PureOp::Zext(to)) => Op::Mach(MachOp::X86Movzx {
                from: from_ty,
                to: to.clone(),
            }),
            Op::Pure(PureOp::Trunc(to)) => Op::Mach(MachOp::X86Trunc {
                from: from_ty,
                to: to.clone(),
            }),
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
        let Op::Pure(PureOp::Bitcast(to)) = &snap.op else {
            continue;
        };

        let child = snap.children[0];
        let Some(from_ty) = infer_class_type(egraph, child) else {
            continue;
        };

        let machine_node = egraph.add(ENode {
            op: Op::Mach(MachOp::X86Bitcast {
                from: from_ty,
                to: to.clone(),
            }),
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
            Op::Pure(PureOp::Fadd) if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::Mach(MachOp::X86Addss), 2)
                } else {
                    (Op::Mach(MachOp::X86Addsd), 2)
                }
            }
            Op::Pure(PureOp::Fsub) if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::Mach(MachOp::X86Subss), 2)
                } else {
                    (Op::Mach(MachOp::X86Subsd), 2)
                }
            }
            Op::Pure(PureOp::Fmul) if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::Mach(MachOp::X86Mulss), 2)
                } else {
                    (Op::Mach(MachOp::X86Mulsd), 2)
                }
            }
            Op::Pure(PureOp::Fdiv) if snap.children.len() == 2 => {
                if is_f32 {
                    (Op::Mach(MachOp::X86Divss), 2)
                } else {
                    (Op::Mach(MachOp::X86Divsd), 2)
                }
            }
            Op::Pure(PureOp::Fsqrt) if snap.children.len() == 1 => {
                if is_f32 {
                    (Op::Mach(MachOp::X86Sqrtss), 1)
                } else {
                    (Op::Mach(MachOp::X86Sqrtsd), 1)
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

        let mut child = snap.children[0];

        let machine_op = match &snap.op {
            Op::Pure(PureOp::IntToFloat(target)) => {
                // cvtsi2sd/ss encode a 32- or 64-bit source only. Anything
                // narrower has to be widened first: the high bits of a register
                // holding an i8/i16 are undefined, and so is the high half of
                // one holding an i32, so picking the wrong width converts
                // whatever the caller happened to leave behind.
                let src_ty = match infer_class_type(egraph, child) {
                    Some(Type::I64) => Type::I64,
                    Some(Type::I32) | None => Type::I32,
                    Some(narrow) => {
                        debug_assert!(
                            matches!(narrow, Type::I8 | Type::I16),
                            "unexpected integer source {narrow:?} for IntToFloat"
                        );
                        // Emit the machine op, not `Sext`: this runs inside
                        // isel, so a generic node added here would still need
                        // its own lowering alternative and extraction would
                        // find the class has none.
                        child = egraph.add(ENode {
                            op: Op::Mach(MachOp::X86Movsx {
                                from: narrow,
                                to: Type::I32,
                            }),
                            children: smallvec![child],
                        });
                        Type::I32
                    }
                };
                match target {
                    Type::F64 => Op::Mach(MachOp::X86Cvtsi2sd(src_ty)),
                    Type::F32 => Op::Mach(MachOp::X86Cvtsi2ss(src_ty)),
                    other => {
                        unreachable!("IntToFloat target must be F32 or F64, got {:?}", other);
                    }
                }
            }
            Op::Pure(PureOp::FloatToInt(target)) => {
                let child_ty = infer_class_type(egraph, child);
                let is_f32 = matches!(child_ty, Some(Type::F32));
                if is_f32 {
                    Op::Mach(MachOp::X86Cvttss2si(target.clone()))
                } else {
                    Op::Mach(MachOp::X86Cvttsd2si(target.clone()))
                }
            }
            Op::Pure(PureOp::FloatExt) => Op::Mach(MachOp::X86Cvtss2sd),
            Op::Pure(PureOp::FloatTrunc) => Op::Mach(MachOp::X86Cvtsd2ss),
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
        if let Op::Pure(PureOp::Icmp(cc)) = &node.op {
            return Some(*cc);
        }
        if let Op::Pure(PureOp::Fcmp(cc)) = &node.op {
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
            op: Op::Pure(PureOp::Iconst(v, Type::I64)),
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
            op: Op::Pure(PureOp::Add),
            children: smallvec![a, b],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let x86add = g.add(ENode {
            op: Op::Mach(MachOp::X86Add),
            children: smallvec![a, b],
        });
        let proj0 = g.add(ENode {
            op: Op::Pure(PureOp::Proj0),
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
            op: Op::Pure(PureOp::Sub),
            children: smallvec![a, b],
        });
        let icmp = g.add(ENode {
            op: Op::Pure(PureOp::Icmp(CondCode::Slt)),
            children: smallvec![a, b],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        // Both should reference the same X86Sub
        let x86sub = g.add(ENode {
            op: Op::Mach(MachOp::X86Sub),
            children: smallvec![a, b],
        });
        let proj0 = g.add(ENode {
            op: Op::Pure(PureOp::Proj0),
            children: smallvec![x86sub],
        });
        let proj1 = g.add(ENode {
            op: Op::Pure(PureOp::Proj1),
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
            op: Op::Pure(PureOp::Icmp(CondCode::Slt)),
            children: smallvec![a, b],
        });
        let icmp_ult = g.add(ENode {
            op: Op::Pure(PureOp::Icmp(CondCode::Ult)),
            children: smallvec![a, b],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let x86sub = g.add(ENode {
            op: Op::Mach(MachOp::X86Sub),
            children: smallvec![a, b],
        });
        let proj1 = g.add(ENode {
            op: Op::Pure(PureOp::Proj1),
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
            op: Op::Pure(PureOp::Icmp(CondCode::Eq)),
            children: smallvec![a, b],
        });
        let t = iconst(&mut g, 1);
        let f = iconst(&mut g, 0);
        let sel = g.add(ENode {
            op: Op::Pure(PureOp::Select),
            children: smallvec![flags, t, f],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let cmov = g.add(ENode {
            op: Op::Mach(MachOp::X86Cmov(CondCode::Eq)),
            children: smallvec![flags, t, f],
        });
        assert_eq!(g.find(sel), g.find(cmov));
    }

    // Sext(I64) on an I32 value merges with X86Movsx{I32, I64}
    #[test]
    fn sext_i32_to_i64_isel() {
        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(42, Type::I32)),
            children: smallvec![],
        });
        let sext = g.add(ENode {
            op: Op::Pure(PureOp::Sext(Type::I64)),
            children: smallvec![val],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let movsx = g.add(ENode {
            op: Op::Mach(MachOp::X86Movsx {
                from: Type::I32,
                to: Type::I64,
            }),
            children: smallvec![val],
        });
        assert_eq!(g.find(sext), g.find(movsx));
    }

    // Zext(I64) on an I8 value merges with X86Movzx{I8, I64}
    #[test]
    fn zext_i8_to_i64_isel() {
        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(1, Type::I8)),
            children: smallvec![],
        });
        let zext = g.add(ENode {
            op: Op::Pure(PureOp::Zext(Type::I64)),
            children: smallvec![val],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let movzx = g.add(ENode {
            op: Op::Mach(MachOp::X86Movzx {
                from: Type::I8,
                to: Type::I64,
            }),
            children: smallvec![val],
        });
        assert_eq!(g.find(zext), g.find(movzx));
    }

    // Trunc(I32) on an I64 value merges with X86Trunc{I64, I32}
    #[test]
    fn trunc_i64_to_i32_isel() {
        let mut g = EGraph::new();
        let val = g.add(ENode {
            op: Op::Pure(PureOp::Iconst(0xFF_FFFF_FFFFi64, Type::I64)),
            children: smallvec![],
        });
        let trunc = g.add(ENode {
            op: Op::Pure(PureOp::Trunc(Type::I32)),
            children: smallvec![val],
        });
        apply_isel_rules(&mut g);
        g.rebuild();

        let x86trunc = g.add(ENode {
            op: Op::Mach(MachOp::X86Trunc {
                from: Type::I64,
                to: Type::I32,
            }),
            children: smallvec![val],
        });
        assert_eq!(g.find(trunc), g.find(x86trunc));
    }
}
