use std::collections::BTreeSet;

use crate::x86::inst::{LabelId, MachInst};

/// Insert multi-byte NOPs before loop headers to align them to 16-byte boundaries.
///
/// `loop_headers` is the set of label IDs that are loop headers.
/// `inst_sizes` returns the encoded byte size of a given instruction.
///
/// The instruction list is expected to contain `MachInst::Label` markers that
/// record where labels are defined. Since `MachInst` does not have a `Label`
/// variant, callers must use a parallel structure or a wrapper. In this
/// implementation the function uses the `Label` instructions from the
/// instruction stream to track positions.
///
/// Because `MachInst` has no `Label` variant, label positions are passed in
/// separately via `label_defs`: a slice of `(inst_idx, LabelId)` pairs
/// indicating that `label` is defined just before `insts[inst_idx]`.
///
/// NOP alignment MUST run before branch relaxation.
pub fn align_loop_headers(
    insts: &mut Vec<MachInst>,
    label_defs: &[(usize, LabelId)],
    loop_headers: &BTreeSet<LabelId>,
    inst_sizes: &dyn Fn(&MachInst) -> usize,
) {
    if loop_headers.is_empty() {
        return;
    }

    // Build a sorted list of (inst_idx, label_id) for loop headers only.
    let mut sites: Vec<(usize, LabelId)> = label_defs
        .iter()
        .filter(|(_, lid)| loop_headers.contains(lid))
        .copied()
        .collect();
    // Process in reverse order so that inserting NOPs before an earlier site
    // does not shift the indices of later sites.
    sites.sort_by_key(|&(idx, _)| idx);
    sites.reverse();

    for (inst_idx, _label) in sites {
        // Compute the byte offset at inst_idx by summing sizes of all instructions before it.
        let offset: usize = insts[..inst_idx].iter().map(inst_sizes).sum();
        let misalign = offset % 16;
        if misalign == 0 {
            continue; // already aligned
        }
        let pad = 16 - misalign;
        // Insert a single multi-byte NOP of `pad` bytes (1..=15).
        debug_assert!(pad <= 15, "NOP pad size must be 1-15");
        insts.insert(inst_idx, MachInst::Nop { size: pad as u8 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x86::inst::MachInst;

    /// Toy size function: each instruction is 4 bytes except Nop (its size field).
    fn size(inst: &MachInst) -> usize {
        match inst {
            MachInst::Nop { size } => *size as usize,
            _ => 4,
        }
    }

    #[test]
    fn loop_header_at_aligned_offset_unchanged() {
        // 4 instructions * 4 bytes = 16-byte offset -> already aligned.
        let mut insts: Vec<MachInst> = vec![
            MachInst::Ret, // 0..4
            MachInst::Ret, // 4..8
            MachInst::Ret, // 8..12
            MachInst::Ret, // 12..16
            MachInst::Ret, // <- label here at offset 16
        ];
        let label_id: LabelId = 0;
        let label_defs = [(4, label_id)];
        let mut headers = BTreeSet::new();
        headers.insert(label_id);

        align_loop_headers(&mut insts, &label_defs, &headers, &size);
        assert_eq!(
            insts.len(),
            5,
            "no NOP should be inserted when already aligned"
        );
    }

    #[test]
    fn loop_header_at_non_aligned_offset_padded() {
        // 6 instructions * 4 bytes = 24 bytes (offset 0x18). Nearest 16-boundary
        // above is 0x20 (32). Pad = 32 - 24 = 8 bytes.
        let mut insts: Vec<MachInst> = vec![
            MachInst::Ret, // 0
            MachInst::Ret, // 4
            MachInst::Ret, // 8
            MachInst::Ret, // 12
            MachInst::Ret, // 16
            MachInst::Ret, // 20
            MachInst::Ret, // <- label at offset 24 (0x18)
        ];
        let label_id: LabelId = 1;
        let label_defs = [(6, label_id)];
        let mut headers = BTreeSet::new();
        headers.insert(label_id);

        align_loop_headers(&mut insts, &label_defs, &headers, &size);

        // A NOP should have been inserted at position 6.
        assert_eq!(insts.len(), 8);
        assert_eq!(insts[6], MachInst::Nop { size: 8 });

        // Verify resulting offset is 16-byte aligned.
        let offset: usize = insts[..7].iter().map(size).sum();
        assert_eq!(offset % 16, 0);
    }

    #[test]
    fn spec_example_offset_0x1a() {
        // Loop header at offset 0x1A (26). Pad = 32 - 26 = 6 bytes.
        // Construct: 6 * 4 = 24 bytes, then 1 * 2 bytes = 26 total.
        // Use a Nop{2} as the last instruction before the label.
        let mut insts: Vec<MachInst> = vec![
            MachInst::Ret,             // 4
            MachInst::Ret,             // 8
            MachInst::Ret,             // 12
            MachInst::Ret,             // 16
            MachInst::Ret,             // 20
            MachInst::Ret,             // 24
            MachInst::Nop { size: 2 }, // 26 = 0x1A
            MachInst::Ret,             // <- label here at offset 26
        ];
        let label_id: LabelId = 2;
        let label_defs = [(7, label_id)];
        let mut headers = BTreeSet::new();
        headers.insert(label_id);

        align_loop_headers(&mut insts, &label_defs, &headers, &size);

        assert_eq!(insts[7], MachInst::Nop { size: 6 });

        let offset: usize = insts[..8].iter().map(size).sum();
        assert_eq!(offset, 32); // 0x20
        assert_eq!(offset % 16, 0);
    }
}
