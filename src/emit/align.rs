use std::collections::{BTreeMap, BTreeSet};

use crate::x86::inst::{LabelId, MachInst};

/// Loop headers are padded to this boundary.
const LOOP_ALIGN: usize = 16;

/// The largest padding worth paying for, in the max-skip form of gcc's
/// `-falign-loops`. At 15 nothing is ever skipped and every header is aligned.
///
/// **Capping this was measured twice and the answer inverted.** Against the
/// code blitz emitted before optimistic coalescing, capping at 8 aligned 54% of
/// headers and measured `x2.797` cycles where no cap aligned 99% and measured
/// `x2.819` -- so the cap won, on half the bytes. Against the code with 28%
/// fewer copies it loses: `x2.725` capped against `x2.685` uncapped, over two
/// samples each that reproduce to 0.15%.
///
/// The lesson is about the measurement, not the constant. **A padding budget is
/// priced against the code it pads**, and 7% fewer instructions is enough to
/// move where the trade lands -- so this is not a number to carry forward
/// through a change that moves code size. Re-measure it.
pub const MAX_SKIP: usize = 15;

/// How many align/relax rounds to run before taking what the last one produced.
/// Each round only moves an offset when relaxation widened a jump, and widening
/// is monotone, so the sequence settles quickly or not at all.
const MAX_ROUNDS: usize = 8;

/// The labels a backward jump targets: a loop header, as the emitted stream
/// sees it.
///
/// Nothing here consults the CFG. `cfg::block_loop_depths` is keyed on
/// `func.blocks` order, blocks are emitted in RPO, and a trampoline label can
/// be a back-edge target without being a block at all. A jump whose target
/// sits at or before the jump itself is the whole definition.
pub fn loop_header_labels(
    insts: &[MachInst],
    label_positions: &BTreeMap<LabelId, usize>,
) -> BTreeSet<LabelId> {
    let mut headers = BTreeSet::new();
    for (i, inst) in insts.iter().enumerate() {
        let target = match inst {
            MachInst::Jmp { target } => *target,
            MachInst::Jcc { target, .. } => *target,
            _ => continue,
        };
        if label_positions.get(&target).is_some_and(|&pos| pos <= i) {
            headers.insert(target);
        }
    }
    headers
}

/// Byte offset of every instruction, given which jumps are short.
fn offsets(
    insts: &[MachInst],
    is_short: &[bool],
    start: usize,
    inst_sizes: &dyn Fn(&MachInst) -> usize,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(insts.len() + 1);
    let mut cur = start;
    for (i, inst) in insts.iter().enumerate() {
        out.push(cur);
        cur += super::relax::branch_size(inst, is_short[i], inst_sizes);
    }
    out.push(cur);
    out
}

/// Padding to insert before each loop header, so that the header lands on a
/// 16-byte boundary in the *encoded* function.
///
/// `prologue_size` is the distance from the function's first byte to the first
/// instruction of `insts`. The prologue is encoded separately from the
/// instruction stream and is part of every absolute offset; a function's own
/// first byte is 16-byte aligned by `compile::module`, which is what makes the
/// answer stable at all.
///
/// The result maps a header label to the number of NOP bytes that must precede
/// its binding point. Only labels in the map are padded, and a pad is never
/// larger than [`MAX_SKIP`].
///
/// **Padding and relaxation each move the other**, so this iterates them.
/// Inserting bytes can push a jump out of rel8 range, which widens it by 3 or 4
/// and moves every header after it; re-padding for the new offsets can in turn
/// widen another jump. The caller must relax the *final* padded stream --
/// correctness does not rest on this converging, only the quality of the
/// alignment does.
pub fn loop_header_pads(
    insts: &[MachInst],
    label_positions: &BTreeMap<LabelId, usize>,
    headers: &BTreeSet<LabelId>,
    prologue_size: usize,
    inst_sizes: &dyn Fn(&MachInst) -> usize,
) -> BTreeMap<LabelId, u8> {
    if headers.is_empty() {
        return BTreeMap::new();
    }

    // One header per binding point. Two labels can bind at the same instruction
    // -- a trampoline label at the top of a block shares its position with the
    // block label -- and padding both would pad twice for one boundary.
    let mut sites: BTreeMap<usize, LabelId> = BTreeMap::new();
    for &label in headers {
        if let Some(&pos) = label_positions.get(&label) {
            sites.entry(pos).or_insert(label);
        }
    }

    let mut pads: BTreeMap<LabelId, u8> = BTreeMap::new();
    for _ in 0..MAX_ROUNDS {
        let padded = apply_pads(insts, label_positions, &pads);
        let padded_positions = shift_positions(label_positions, insts.len(), &pads);
        let (_, is_short) = super::relax::relax_branches(&padded, &padded_positions, inst_sizes);
        let offs = offsets(&padded, &is_short, prologue_size, inst_sizes);

        let mut next: BTreeMap<LabelId, u8> = BTreeMap::new();
        for &label in sites.values() {
            // This offset already includes the label's own padding, which is in
            // `padded`. Take it back off to ask where the header would sit
            // unpadded, or a header that is aligned *because* it was padded
            // reads as needing nothing and loses its pad next round.
            let current = pads.get(&label).copied().unwrap_or(0) as usize;
            let bare = offs[padded_positions[&label]] - current;
            let want = (LOOP_ALIGN - bare % LOOP_ALIGN) % LOOP_ALIGN;
            if want > 0 && want <= MAX_SKIP {
                next.insert(label, want as u8);
            }
        }

        if next == pads {
            break;
        }
        pads = next;
    }

    pads
}

/// The instruction stream with each pad inserted immediately before the
/// instruction its header label binds to.
pub fn apply_pads(
    insts: &[MachInst],
    label_positions: &BTreeMap<LabelId, usize>,
    pads: &BTreeMap<LabelId, u8>,
) -> Vec<MachInst> {
    if pads.is_empty() {
        return insts.to_vec();
    }
    let mut at: BTreeMap<usize, u8> = BTreeMap::new();
    for (label, &pad) in pads {
        let pos = label_positions[label];
        at.insert(pos, pad);
    }
    let mut out = Vec::with_capacity(insts.len() + at.len());
    for (i, inst) in insts.iter().enumerate() {
        if let Some(&pad) = at.get(&i) {
            out.push(MachInst::Nop { size: pad });
        }
        out.push(inst.clone());
    }
    if let Some(&pad) = at.get(&insts.len()) {
        out.push(MachInst::Nop { size: pad });
    }
    out
}

/// Label positions after [`apply_pads`] inserted its NOPs.
///
/// Every label binds *after* the NOPs inserted at or before its position: the
/// padding belongs to the boundary, so a label sharing a padded position with
/// the header that owns the pad moves with it.
pub fn shift_positions(
    label_positions: &BTreeMap<LabelId, usize>,
    inst_count: usize,
    pads: &BTreeMap<LabelId, u8>,
) -> BTreeMap<LabelId, usize> {
    if pads.is_empty() {
        return label_positions.clone();
    }
    let padded_at: BTreeSet<usize> = pads.keys().map(|l| label_positions[l]).collect();
    // inserted[i] = how many NOPs precede original instruction index i.
    let mut inserted = vec![0usize; inst_count + 1];
    let mut running = 0usize;
    for (i, slot) in inserted.iter_mut().enumerate() {
        if padded_at.contains(&i) {
            running += 1;
        }
        *slot = running;
    }
    label_positions
        .iter()
        .map(|(&label, &pos)| (label, pos + inserted[pos]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::condcode::CondCode;

    /// Toy size function: each instruction is 4 bytes except Nop (its size field).
    fn size(inst: &MachInst) -> usize {
        match inst {
            MachInst::Nop { size } => *size as usize,
            _ => 4,
        }
    }

    fn positions(pairs: &[(LabelId, usize)]) -> BTreeMap<LabelId, usize> {
        pairs.iter().copied().collect()
    }

    /// `total` filler instructions, then a backward jump to `label`.
    fn loop_at(label: LabelId, total: usize) -> Vec<MachInst> {
        let mut insts = vec![MachInst::Ret; total];
        insts.push(MachInst::Jcc {
            cc: CondCode::Ne,
            target: label,
        });
        insts
    }

    #[test]
    fn backward_jump_target_is_a_header() {
        let insts = loop_at(7, 4);
        let pos = positions(&[(7, 2)]);
        let headers = loop_header_labels(&insts, &pos);
        assert_eq!(headers.iter().copied().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn forward_jump_target_is_not_a_header() {
        let insts = vec![
            MachInst::Jmp { target: 3 },
            MachInst::Ret,
            MachInst::Ret,
            MachInst::Ret,
        ];
        let pos = positions(&[(3, 3)]);
        assert!(loop_header_labels(&insts, &pos).is_empty());
    }

    #[test]
    fn header_already_aligned_gets_no_pad() {
        // Prologue 0, four 4-byte instructions, header at 16.
        let insts = loop_at(1, 8);
        let pos = positions(&[(1, 4)]);
        let headers = loop_header_labels(&insts, &pos);
        let pads = loop_header_pads(&insts, &pos, &headers, 0, &size);
        assert!(pads.is_empty(), "got {pads:?}");
    }

    #[test]
    fn prologue_is_part_of_the_offset() {
        // Header at instruction index 4, which is 16 bytes into the stream: it
        // needs nothing with no prologue, and the prologue's own length with one.
        let insts = loop_at(1, 8);
        let pos = positions(&[(1, 4)]);
        let headers = loop_header_labels(&insts, &pos);
        assert!(loop_header_pads(&insts, &pos, &headers, 0, &size).is_empty());

        // A prologue of 8 leaves the header at 24, wanting 8.
        let pads = loop_header_pads(&insts, &pos, &headers, 8, &size);
        assert_eq!(pads.get(&1), Some(&8), "got {pads:?}");

        // And of 4, at 20, wanting 12.
        let pads = loop_header_pads(&insts, &pos, &headers, 4, &size);
        assert_eq!(pads.get(&1), Some(&12), "got {pads:?}");
    }

    #[test]
    fn no_pad_exceeds_max_skip() {
        // Every distance from a boundary, so the widest pad the rule can want is
        // among them.
        let insts = loop_at(1, 8);
        let pos = positions(&[(1, 4)]);
        let headers = loop_header_labels(&insts, &pos);
        for prologue in 0..64 {
            for &pad in loop_header_pads(&insts, &pos, &headers, prologue, &size).values() {
                assert!(pad as usize <= MAX_SKIP, "pad {pad} at prologue {prologue}");
            }
        }
    }

    #[test]
    fn padding_lands_the_header_on_a_boundary() {
        // Prologue 12, header at instruction index 5 -> 12 + 20 = 32. Aligned.
        // Move it: prologue 10 -> 30, wants 2.
        let insts = loop_at(1, 10);
        let pos = positions(&[(1, 5)]);
        let headers = loop_header_labels(&insts, &pos);
        let pads = loop_header_pads(&insts, &pos, &headers, 10, &size);
        assert_eq!(pads.get(&1), Some(&2));

        // Verify by re-encoding: the padded stream puts the label at 32.
        let padded = apply_pads(&insts, &pos, &pads);
        let padded_pos = shift_positions(&pos, insts.len(), &pads);
        let (_, is_short) = super::super::relax::relax_branches(&padded, &padded_pos, &size);
        let offs = offsets(&padded, &is_short, 10, &size);
        assert!(offs[padded_pos[&1]].is_multiple_of(LOOP_ALIGN));
    }

    #[test]
    fn two_labels_at_one_position_are_padded_once() {
        let insts = loop_at(1, 10);
        let mut pos = positions(&[(1, 5)]);
        pos.insert(2, 5);
        let mut headers = BTreeSet::new();
        headers.insert(1);
        headers.insert(2);
        let pads = loop_header_pads(&insts, &pos, &headers, 10, &size);
        assert_eq!(pads.len(), 1, "one boundary, one pad: got {pads:?}");
    }

    #[test]
    fn relaxation_after_padding_is_accounted_for() {
        // A conditional jump whose target is just inside rel8 range, followed by
        // a loop header. Padding pushes the target out of range, the jump widens
        // by 4, and the header must still come out aligned.
        let label_fwd: LabelId = 20;
        let label_hdr: LabelId = 21;

        let mut insts = vec![MachInst::Jcc {
            cc: CondCode::Eq,
            target: label_fwd,
        }];
        insts.extend(vec![MachInst::Ret; 31]);
        // Forward target lands here.
        insts.push(MachInst::Ret);
        // Loop header a little further on, with a back edge to it.
        insts.extend(vec![MachInst::Ret; 3]);
        let hdr_idx = insts.len();
        insts.push(MachInst::Ret);
        insts.push(MachInst::Jcc {
            cc: CondCode::Ne,
            target: label_hdr,
        });

        let pos = positions(&[(label_fwd, 32), (label_hdr, hdr_idx)]);
        let headers = loop_header_labels(&insts, &pos);
        assert!(headers.contains(&label_hdr));

        let pads = loop_header_pads(&insts, &pos, &headers, 6, &size);
        let padded = apply_pads(&insts, &pos, &pads);
        let padded_pos = shift_positions(&pos, insts.len(), &pads);
        let (_, is_short) = super::super::relax::relax_branches(&padded, &padded_pos, &size);
        let offs = offsets(&padded, &is_short, 6, &size);
        let landed = offs[padded_pos[&label_hdr]];
        assert!(
            landed.is_multiple_of(LOOP_ALIGN) || pads.is_empty(),
            "header at {landed}, pads {pads:?}"
        );
    }
}
