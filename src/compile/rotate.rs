//! Duplicate a loop's test onto its back edge.
//!
//! A loop laid out header-first closes on an unconditional `jmp`: the header
//! tests and falls through into the body, so the branch that is *not* taken
//! leaves the loop and the one that *is* taken is the back edge. Copying the
//! header's test onto the back edge and inverting its condition makes the back
//! edge itself the conditional, and the header is left as a guard the loop
//! enters through once.
//!
//! **It is not a second taken branch that this removes.** Both shapes execute
//! exactly one taken branch per iteration; what goes away is the unconditional
//! `jmp` and the not-taken conditional above it, so the win is one instruction
//! and one branch and shows up only where the front end is what the loop is
//! bound by. Measured in isolation on Zen 3, with identical bodies and only the
//! loop shape differing: `0.0%` on ALU-bound bodies of 4 to 19 instructions,
//! `0.0%` on a serial dependence chain, and `-4.3%` on an 8-instruction
//! byte-copy body -- the last robust across six loop-start alignments.
//!
//! **The copy is exact, not a hoist.** It is placed on the edge that would have
//! reached the header, so the instructions run at the same point in the trace on
//! the same registers, and the path that runs the copy does not also run the
//! original. Nothing has to be proved live: what makes a run ineligible is that
//! it cannot be *written* twice -- a bound label would be bound twice, and an
//! operand naming memory or a symbol is not known to encode the same at a second
//! address.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::function::Function;
use crate::x86::inst::{LabelId, MachInst};

use super::BlockItem;
use super::licm::detect_loops;
use super::terminator::negate_cc;

/// The most instructions a header's test may cost before duplicating it stops
/// paying. The win is one `jmp` per iteration, so a run longer than that trades
/// instructions for nothing on entry and bytes everywhere.
const MAX_TEST_INSTS: usize = 3;

/// True for an instruction that a second copy of encodes to the same bytes at
/// any address and whose effect does not depend on where it sits.
///
/// Register and immediate operands only. A memory operand is excluded because
/// the addressing mode may be RIP-relative and a symbol reference because it
/// carries a relocation, and neither is duplicated by cloning the instruction.
fn is_duplicable(inst: &MachInst) -> bool {
    matches!(
        inst,
        MachInst::CmpRR { .. }
            | MachInst::CmpRI { .. }
            | MachInst::TestRR { .. }
            | MachInst::TestRI { .. }
            | MachInst::MovRR { .. }
            | MachInst::MovRI { .. }
            | MachInst::AddRR { .. }
            | MachInst::AddRI { .. }
            | MachInst::SubRR { .. }
            | MachInst::SubRI { .. }
            | MachInst::AndRR { .. }
            | MachInst::AndRI { .. }
            | MachInst::OrRR { .. }
            | MachInst::OrRI { .. }
            | MachInst::XorRR { .. }
            | MachInst::XorRI { .. }
            | MachInst::Inc { .. }
            | MachInst::Dec { .. }
            | MachInst::Neg { .. }
            | MachInst::Not { .. }
            | MachInst::UcomisdRR { .. }
            | MachInst::UcomissRR { .. }
    )
}

/// A header whose whole content is a duplicable test ending in one conditional.
struct Test {
    insts: Vec<MachInst>,
    cc: crate::ir::condcode::CondCode,
    /// Where the conditional goes when it is taken. Which side of the loop that
    /// is depends on the sense `lower_terminator` chose, so it is not the exit
    /// by construction and the CFG is what says which successor is the body.
    taken: LabelId,
}

/// The header's items as a test, or `None` if they cannot be copied.
///
/// The last item must be the conditional and nothing may follow it: a trailing
/// `Jmp` means the header does not fall through to the block laid out after it,
/// so there is no body to name. The `OrdEq`/`UnordNe` expansions bind their own
/// label and are rejected by the same rule that rejects any other binding.
fn header_test(items: &[BlockItem]) -> Option<Test> {
    let mut insts = Vec::new();
    for item in items {
        match item {
            BlockItem::BindLabel(_) => return None,
            BlockItem::Inst(inst) => insts.push(inst),
        }
    }
    let (last, rest) = insts.split_last()?;
    let &&MachInst::Jcc { cc, target } = last else {
        return None;
    };
    if rest.len() > MAX_TEST_INSTS || !rest.iter().all(|inst| is_duplicable(inst)) {
        return None;
    }
    Some(Test {
        insts: rest.iter().map(|&inst| inst.clone()).collect(),
        cc,
        taken: target,
    })
}

/// Rotate every loop whose header is a duplicable test reached by a backward
/// `jmp`, and return each rotated header's label paired with the label that is
/// now the loop's entry.
///
/// The mapping is what loop-header alignment must be told: after the rotation
/// the header runs once as a guard and the block laid out after it is what
/// every iteration branches to, so padding the header pads a block outside the
/// loop and leaves the loop wherever it fell.
pub(super) fn rotate_loops(
    block_items: &mut [Vec<BlockItem>],
    func: &Function,
    layout_order: &[usize],
) -> BTreeMap<LabelId, LabelId> {
    let label_at: Vec<LabelId> = layout_order
        .iter()
        .map(|&idx| func.blocks[idx].id as LabelId)
        .collect();
    let mut layout_pos = vec![usize::MAX; func.blocks.len()];
    for (pos, &idx) in layout_order.iter().enumerate() {
        layout_pos[idx] = pos;
    }

    let mut rotated = BTreeMap::new();
    for info in detect_loops(func) {
        let header_pos = layout_pos[info.header_idx];
        if header_pos == usize::MAX {
            continue;
        }
        let header = label_at[header_pos];
        let Some(test) = header_test(&block_items[header_pos]) else {
            continue;
        };
        // The header's two successors are the conditional's target and the
        // block laid out next. Which one continues the loop is the CFG's answer
        // and not the layout's: `lower_terminator` inverts the condition
        // whenever that lets the true side fall through, so either successor can
        // be the body.
        let Some(&fallthrough) = label_at.get(header_pos + 1) else {
            continue;
        };
        let inside: BTreeSet<LabelId> = info
            .body
            .iter()
            .map(|&idx| func.blocks[idx].id as LabelId)
            .filter(|&id| id != header)
            .collect();
        let (body, exit) = match (inside.contains(&test.taken), inside.contains(&fallthrough)) {
            (true, false) => (test.taken, fallthrough),
            (false, true) => (fallthrough, test.taken),
            // Both or neither: the header does not decide this loop's exit, so
            // there is no test to move to the back edge.
            _ => continue,
        };
        // The branch back to the body carries the sense that reaches it, so the
        // conditional is the taken edge either way round.
        let cc = if body == test.taken {
            test.cc
        } else {
            negate_cc(test.cc)
        };

        // Every backward `jmp` to the header, which is every latch: laid out
        // after the header and ending in a jump to it.
        let mut any = false;
        for items in block_items[header_pos + 1..].iter_mut() {
            let ends_in_back_edge = matches!(
                items.last(),
                Some(BlockItem::Inst(MachInst::Jmp { target })) if *target == header
            );
            if !ends_in_back_edge {
                continue;
            }
            items.pop();
            items.extend(test.insts.iter().cloned().map(BlockItem::Inst));
            items.push(BlockItem::Inst(MachInst::Jcc { cc, target: body }));
            // Reached when the loop is done. `remove_fallthrough_jumps` deletes
            // it whenever the exit is what follows, which the trace layout makes
            // the common case.
            items.push(BlockItem::Inst(MachInst::Jmp { target: exit }));
            any = true;
        }
        if any {
            rotated.insert(header, body);
        }
    }
    rotated
}
