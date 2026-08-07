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
fn is_pure_value_move(inst: &MachInst) -> bool {
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
        } => *d != Reg::RSP && *d != Reg::RBP,
        _ => false,
    }
}

/// Indices into `insts` of instructions that compute a value nothing reads.
///
/// Iterated to a fixpoint: removing one can make the instruction that fed it
/// dead in turn, which is the common case here -- a `lea` feeding a `mov` that
/// feeds nothing.
pub fn dead_value_moves(insts: &[MachInst], labels: &BTreeMap<LabelId, usize>) -> BTreeSet<usize> {
    let mut dead: BTreeSet<usize> = BTreeSet::new();
    loop {
        let found = dead_value_moves_round(insts, labels, &dead);
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
            // A block with no successors returns or tail-calls, and either way the
            // ABI's return register and everything the callee reads is live out.
            // Being wrong in this direction only keeps instructions.
            if succs[b].is_empty() {
                live.extend(crate::x86::abi::CALLER_SAVED_GPR);
                live.extend(crate::x86::abi::CALLEE_SAVED);
                live.extend(crate::x86::abi::CALLER_SAVED_XMM);
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
            live.extend(crate::x86::abi::CALLER_SAVED_GPR);
            live.extend(crate::x86::abi::CALLEE_SAVED);
            live.extend(crate::x86::abi::CALLER_SAVED_XMM);
        }
        for i in blocks[b].clone().rev() {
            if already.contains(&i) {
                continue;
            }
            let defs = insts[i].defs();
            if is_pure_value_move(&insts[i]) && defs.iter().all(|r| !live.contains(r)) {
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
