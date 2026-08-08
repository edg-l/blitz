//! Instructions whose result nothing reads, removed from the final stream.
//!
//! **Why this exists at machine level and not earlier.** Lowering folds an
//! address computation into the addressing mode of the load or store that uses
//! it -- `[base + idx*4]` rather than a separate `lea` -- and it decides that per
//! consumer, at the last possible moment. When every consumer folds, the `lea` is
//! still emitted and nothing reads it. Measured over `bench` and `live` before
//! this pass: 122 of 7801 instructions were a `lea` immediately followed by an
//! instruction naming the same address in its own addressing mode, in every one of
//! the 34 kernels. They sit in loop bodies, so their dynamic weight is far above
//! the 1.56% they are statically.
//!
//! No earlier pass can see them. DCE runs on the CFG before scheduling, the
//! e-graph never sees an effectful op, and the folding that makes the `lea` dead
//! happens after both.
//!
//! **What it will delete, and why the list is short.** Only instructions that
//! compute a value into a register and do nothing else: register-to-register
//! moves, immediate loads, and `lea`. Not arithmetic, because EFLAGS is not a
//! register this liveness models and deleting a flag write that something later
//! reads would be silent and wrong. Not anything touching memory, so nothing has
//! to reason about aliasing or faults. Not a write to RSP or RBP, which the frame
//! layout owns rather than any value.
//!
//! Liveness is a backward dataflow over the CFG recovered from labels and
//! branches, the same recovery [`crate::verify`] uses for its machine-level
//! checks -- shared rather than re-derived, because two recoveries of one CFG that
//! disagree is how a dead-code pass deletes something live.

use std::collections::{BTreeMap, BTreeSet};

use crate::x86::inst::{LabelId, MachInst, Operand};
use crate::x86::reg::Reg;

/// Whether this instruction may be removed when its results are dead.
///
/// The question is not "does it write a register" but "is a register write all it
/// does". A `mov` to memory, a call, a branch and an `add` all fail that for
/// different reasons; the last one because it writes EFLAGS, which is not in the
/// liveness below.
///
/// `RSP` is never a value this may drop, and `RBP` is not one either *while the
/// frame pointer is in use* -- there the frame layout owns the register rather
/// than any value, and its writes are structural. With the frame pointer
/// omitted, which is the default, `RBP` is an ordinary allocatable register and
/// excluding it only keeps dead code.
fn is_pure_value_move(inst: &MachInst, frame_pointer: bool) -> bool {
    match inst {
        MachInst::MovRR {
            dst: Operand::Reg(d),
            src: Operand::Reg(_),
            ..
        }
        | MachInst::MovRI {
            dst: Operand::Reg(d),
            ..
        }
        | MachInst::Lea {
            dst: Operand::Reg(d),
            ..
        }
        | MachInst::LeaRipRelative {
            dst: Operand::Reg(d),
            ..
        }
        | MachInst::MovsdRR {
            dst: Operand::Reg(d),
            src: Operand::Reg(_),
        }
        | MachInst::MovssRR {
            dst: Operand::Reg(d),
            src: Operand::Reg(_),
        } => *d != Reg::RSP && !(frame_pointer && *d == Reg::RBP),
        _ => false,
    }
}

/// Indices into `insts` of instructions that compute a value nothing reads.
///
/// Iterated to a fixpoint: removing one can make the instruction that fed it
/// dead in turn, which is the common case here -- a `lea` feeding a `mov` that
/// feeds nothing.
pub fn dead_value_moves(
    insts: &[MachInst],
    labels: &BTreeMap<LabelId, usize>,
    frame_pointer: bool,
) -> BTreeSet<usize> {
    let mut dead: BTreeSet<usize> = BTreeSet::new();
    loop {
        let found = dead_value_moves_round(insts, labels, &dead, frame_pointer);
        if found.is_empty() {
            return dead;
        }
        dead.extend(found);
    }
}

fn dead_value_moves_round(
    insts: &[MachInst],
    labels: &BTreeMap<LabelId, usize>,
    already: &BTreeSet<usize>,
    frame_pointer: bool,
) -> BTreeSet<usize> {
    let leaders = crate::verify::block_leaders(insts, labels);
    let (blocks, succs) = crate::verify::build_cfg(insts, labels, &leaders);

    // Backward liveness over physical registers. A register is live at a point if
    // some path from there reads it before writing it, so successors are met with
    // union -- the opposite of the intersection the def-before-use check uses,
    // and for the opposite reason: that one asks what is guaranteed, this asks
    // what is possible.
    let n = blocks.len();
    let mut live_in: Vec<BTreeSet<Reg>> = vec![BTreeSet::new(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut live: BTreeSet<Reg> = BTreeSet::new();
            for &s in &succs[b] {
                live.extend(live_in[s].iter().copied());
            }
            if succs[b].is_empty() {
                live.extend(exit_live_out(insts, &blocks[b], frame_pointer));
            }
            for i in blocks[b].clone().rev() {
                if already.contains(&i) {
                    continue;
                }
                for r in insts[i].defs() {
                    live.remove(&r);
                }
                live.extend(insts[i].uses());
                live.extend(call_reads(&insts[i]));
            }
            if live_in[b] != live {
                live_in[b] = live;
                changed = true;
            }
        }
    }

    let mut found = BTreeSet::new();
    for b in 0..n {
        let mut live: BTreeSet<Reg> = BTreeSet::new();
        for &s in &succs[b] {
            live.extend(live_in[s].iter().copied());
        }
        if succs[b].is_empty() {
            live.extend(exit_live_out(insts, &blocks[b], frame_pointer));
        }
        for i in blocks[b].clone().rev() {
            if already.contains(&i) {
                continue;
            }
            let defs = insts[i].defs();
            if is_pure_value_move(&insts[i], frame_pointer)
                && defs.iter().all(|r| !live.contains(r))
            {
                found.insert(i);
                // Its operands are not read after all, so do not mark them live.
                continue;
            }
            for r in defs {
                live.remove(&r);
            }
            live.extend(insts[i].uses());
            live.extend(call_reads(&insts[i]));
        }
    }
    found
}

/// What is live where control leaves the function.
///
/// **A callee-saved register is not live at a `ret`.** Its value belongs to the
/// caller, and the epilogue restores it with a `pop` that overwrites whatever
/// this function put there -- so a write to one on a path to a `ret` is dead
/// like any other. Marking them live instead is what kept 47 `lea`s alive
/// across `bench` and `live`, every one of them an address the store beside it
/// already folded into its own addressing mode. They sit in loop bodies.
///
/// A caller-saved register is not live either, for the plainer reason that the
/// caller does not expect one back. What is left is the value being returned --
/// both classes, since the instruction does not say which the function
/// returns -- and `RSP`, which the epilogue reads.
///
/// `GPR_RETURN_REG2` and `XMM1` are *not* here, and that is the one entry with
/// an obligation attached: they carry the second half of a two-register return,
/// which `lower_terminator` never writes -- a `Ret` puts its value in `RAX` or
/// `XMM0` and nothing else. A lowering that starts returning a pair has to add
/// them, or a write to the second register on the way out reads as dead. They
/// cost real deletions in the meantime: `RDX` is the scratch register a scaled
/// index lands in.
///
/// `RBP` joins them only while the frame pointer is in use, since that is what
/// makes the epilogue read it.
///
/// Any other way of leaving keeps the conservative set. A tail call names what
/// it reads through [`call_reads`], but a block ending in something else is a
/// shape this does not model, and being wrong in that direction only keeps
/// instructions.
fn exit_live_out(
    insts: &[MachInst],
    block: &std::ops::Range<usize>,
    frame_pointer: bool,
) -> Vec<Reg> {
    let last = block.end.checked_sub(1).and_then(|i| insts.get(i));
    if matches!(last, Some(MachInst::Ret)) {
        let mut v = vec![
            crate::x86::abi::GPR_RETURN_REG,
            crate::x86::abi::FP_RETURN_REG,
            Reg::RSP,
        ];
        // `mov rsp, rbp; pop rbp` reads RBP; a plain `pop rbp` does not.
        if frame_pointer {
            v.push(Reg::RBP);
        }
        return v;
    }
    let mut v: Vec<Reg> = crate::x86::abi::CALLER_SAVED_GPR.to_vec();
    v.extend(crate::x86::abi::CALLEE_SAVED);
    v.extend(crate::x86::abi::CALLER_SAVED_XMM);
    v
}

/// The registers a call reads, which its `uses()` does not name.
///
/// `MachInst::CallDirect` carries a symbol and no operands, so as far as
/// `uses()` is concerned it reads nothing -- and the `mov`s that put the
/// arguments in their ABI registers are then dead. That mistake deleted the
/// argument setup of every call in the corpus: 248 of 576 lit tests.
///
/// Every argument register, not the ones this particular call happens to use,
/// because the instruction does not say which those are. RAX is included for the
/// vector-count byte SysV puts in AL on every call.
fn call_reads(inst: &MachInst) -> Vec<Reg> {
    if !matches!(
        inst,
        MachInst::CallDirect { .. }
            | MachInst::CallIndirect { .. }
            | MachInst::TailCallDirect { .. }
    ) {
        return Vec::new();
    }
    let mut v: Vec<Reg> = crate::x86::abi::GPR_ARG_REGS.to_vec();
    v.extend(crate::x86::abi::FP_ARG_REGS);
    v.push(Reg::RAX);
    v.push(Reg::RSP);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x86::addr::Addr;
    use crate::x86::inst::OpSize;

    /// `lea dst, [RSP]`, the shape this pass exists to remove.
    fn lea(dst: Reg) -> MachInst {
        MachInst::Lea {
            size: OpSize::S64,
            dst: Operand::Reg(dst),
            addr: Addr::new(Some(Reg::RSP), None, 1, 0),
        }
    }

    fn dead_in(insts: &[MachInst], frame_pointer: bool) -> BTreeSet<usize> {
        dead_value_moves(insts, &BTreeMap::new(), frame_pointer)
    }

    /// The value a callee-saved register holds at a `ret` belongs to the caller,
    /// and the epilogue's `pop` is what puts it back -- so writing one on the way
    /// out is dead.
    #[test]
    fn a_callee_saved_write_before_a_ret_is_dead() {
        assert_eq!(
            dead_in(&[lea(Reg::R15), MachInst::Ret], false),
            BTreeSet::from([0])
        );
    }

    /// The return value is the one thing that does survive.
    #[test]
    fn the_return_register_is_live_at_a_ret() {
        assert!(dead_in(&[lea(Reg::RAX), MachInst::Ret], false).is_empty());
        assert!(dead_in(&[lea(Reg::XMM0), MachInst::Ret], false).is_empty());
    }

    /// A tail call reads the argument registers and leaves through the callee,
    /// which is a shape the exit rule does not model: it keeps everything.
    #[test]
    fn a_tail_call_keeps_the_conservative_set() {
        let insts = [
            lea(Reg::R15),
            MachInst::TailCallDirect {
                target: "g".to_string(),
            },
        ];
        assert!(dead_in(&insts, false).is_empty());
    }

    /// With the frame pointer omitted RBP is an ordinary register; with it in
    /// use the frame layout owns the register and no write to it may go.
    #[test]
    fn rbp_is_a_value_only_while_the_frame_pointer_is_omitted() {
        let insts = [lea(Reg::RBP), MachInst::Ret];
        assert_eq!(dead_in(&insts, false), BTreeSet::from([0]));
        assert!(dead_in(&insts, true).is_empty());
    }

    /// RSP is never a value this may drop, frame pointer or not.
    #[test]
    fn rsp_is_never_a_value() {
        let insts = [lea(Reg::RSP), MachInst::Ret];
        assert!(dead_in(&insts, false).is_empty());
        assert!(dead_in(&insts, true).is_empty());
    }

    /// Removing one write can make the write that fed it dead, which is the
    /// common case: a `lea` into a `mov` into nothing.
    #[test]
    fn a_chain_dies_to_a_fixpoint() {
        let insts = [
            lea(Reg::R15),
            MachInst::MovRR {
                size: OpSize::S64,
                dst: Operand::Reg(Reg::R14),
                src: Operand::Reg(Reg::R15),
            },
            MachInst::Ret,
        ];
        assert_eq!(dead_in(&insts, false), BTreeSet::from([0, 1]));
    }
}
