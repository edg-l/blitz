use std::collections::{BTreeMap, BTreeSet};

use crate::compile::program_point::ProgramPoint;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::ClassId;
use crate::schedule::scheduler::ScheduledInst;

/// Per-block liveness information computed by global iterative dataflow.
pub struct GlobalLiveness {
    /// VRegs live at the start of each block (indexed by block index).
    pub live_in: Vec<BTreeSet<VReg>>,
    /// VRegs live at the end of each block (indexed by block index).
    pub live_out: Vec<BTreeSet<VReg>>,
}

/// Compute per-block live_in and live_out sets using backward iterative dataflow.
///
/// `block_schedules[i]` is the scheduled instruction list for block i.
/// `successors[i]` is the list of block indices that block i can jump to.
/// `phi_uses[i]` is the set of VRegs referenced in block i's terminator
///   (Jump/Branch args) that are not already in the scheduled instruction list.
///
/// Algorithm (standard backward liveness):
///   def(B) = VRegs defined in B's scheduled instructions
///   use(B) = VRegs used (as operands) in B but not defined in B, UNION phi_uses[B]
///   live_out(B) = union over successors S of live_in(S)
///   live_in(B)  = use(B) | (live_out(B) - def(B))
/// Iterate until fixed point.
pub fn compute_global_liveness(
    block_schedules: &[Vec<ScheduledInst>],
    successors: &[Vec<usize>],
    phi_uses: &[BTreeSet<VReg>],
) -> GlobalLiveness {
    compute_global_liveness_with_block_params(block_schedules, successors, phi_uses, &[])
}

/// Same as `compute_global_liveness`, but also treats each entry of
/// `block_param_vregs_per_block[b]` as live at block-B entry. Block params are
/// the destinations of phi copies; the predecessor's phi_copies write to each
/// param at a single program point (block entry), so they must receive distinct
/// physical registers even when the block body never reads them. Adding them to
/// live_in forces pairwise interference via `build_interference_into`'s
/// cross-block boundary edges.
pub fn compute_global_liveness_with_block_params(
    block_schedules: &[Vec<ScheduledInst>],
    successors: &[Vec<usize>],
    phi_uses: &[BTreeSet<VReg>],
    block_param_vregs_per_block: &[BTreeSet<VReg>],
) -> GlobalLiveness {
    let n = block_schedules.len();
    assert_eq!(successors.len(), n);
    assert_eq!(phi_uses.len(), n);

    // Compute def(B) and use(B) for each block.
    let mut block_def: Vec<BTreeSet<VReg>> = Vec::with_capacity(n);
    let mut block_use: Vec<BTreeSet<VReg>> = Vec::with_capacity(n);

    for (b, sched) in block_schedules.iter().enumerate() {
        let mut def: BTreeSet<VReg> = BTreeSet::new();
        let mut uses: BTreeSet<VReg> = BTreeSet::new();

        // Process instructions in forward order to compute upward-exposed uses.
        for inst in sched {
            // Operands that are not yet defined in this block are upward-exposed uses.
            for &op in &inst.operands {
                if !def.contains(&op) {
                    uses.insert(op);
                }
            }
            def.insert(inst.dst);
        }

        // phi_uses[b] are VRegs used in the block's terminator.
        // If they are not defined in this block, they are upward-exposed.
        for &v in &phi_uses[b] {
            if !def.contains(&v) {
                uses.insert(v);
            }
        }

        block_def.push(def);
        block_use.push(uses);
    }

    let mut live_in: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); n];
    let mut live_out: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); n];

    // Initialize live_in = use(B) ∪ block_params(B). Block params receive
    // distinct values from phi copies at block entry and must occupy distinct
    // registers throughout that copy sequence. The simplest way to enforce
    // this in a liveness-driven interference builder is to treat params as
    // "live at entry" so the boundary pass wires up pairwise edges.
    for b in 0..n {
        live_in[b].extend(block_use[b].iter().copied());
        if let Some(params) = block_param_vregs_per_block.get(b) {
            live_in[b].extend(params.iter().copied());
        }
    }

    // Iterate until fixed point.
    let mut changed = true;
    while changed {
        changed = false;
        // Process in reverse order (backward pass heuristic for faster convergence).
        for b in (0..n).rev() {
            // live_out(B) = phi_uses(B) ∪ union of (live_in(S) - block_params(S)).
            //
            // phi_uses[b] is the set of VRegs B's terminator passes to its
            // successors' block params. These values are live at B's end.
            //
            // Successor block params are pre-seeded into live_in[succ] to force
            // pairwise interference among them at the successor's entry. Those
            // VRegs do NOT exist at the predecessor's end (the predecessor
            // terminator writes phi-source VRegs into the successor's params
            // via phi copies), so when propagating live_in[succ] upward we
            // strip out the successor's block params.
            let mut new_out: BTreeSet<VReg> = BTreeSet::new();
            for &v in &phi_uses[b] {
                new_out.insert(v);
            }
            for &s in &successors[b] {
                let succ_params = block_param_vregs_per_block.get(s);
                for &v in &live_in[s] {
                    if let Some(params) = succ_params
                        && params.contains(&v)
                    {
                        continue;
                    }
                    new_out.insert(v);
                }
            }

            if new_out != live_out[b] {
                live_out[b] = new_out;
                changed = true;
            }

            // live_in(B) = use(B) ∪ block_params(B) ∪ (live_out(B) - def(B)).
            let mut new_in = block_use[b].clone();
            if let Some(params) = block_param_vregs_per_block.get(b) {
                new_in.extend(params.iter().copied());
            }
            for &v in &live_out[b] {
                if !block_def[b].contains(&v) {
                    new_in.insert(v);
                }
            }

            if new_in != live_in[b] {
                live_in[b] = new_in;
                changed = true;
            }
        }
    }

    GlobalLiveness { live_in, live_out }
}

/// Extract CFG successor block indices from each block's terminator.
///
/// Returns `successors[i]` = list of block indices that block `i` can jump to.
/// Indices are into `func.blocks` (not block IDs).
pub fn cfg_successors(func: &Function) -> Vec<Vec<usize>> {
    let id_to_idx: BTreeMap<u32, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    func.blocks
        .iter()
        .map(|block| {
            let mut succs = Vec::new();
            if let Some(term) = block.ops.last() {
                match term {
                    EffectfulOp::Jump { target, .. } => {
                        if let Some(&idx) = id_to_idx.get(target) {
                            succs.push(idx);
                        }
                    }
                    EffectfulOp::Branch {
                        bb_true, bb_false, ..
                    } => {
                        if let Some(&idx) = id_to_idx.get(bb_true) {
                            succs.push(idx);
                        }
                        if let Some(&idx) = id_to_idx.get(bb_false) {
                            succs.push(idx);
                        }
                    }
                    _ => {}
                }
            }
            succs
        })
        .collect()
}

/// Collect the set of VRegs that are block parameters for each block.
///
/// Block params are handled by phi elimination and should not be treated as
/// cross-block live-in values that need reload instructions.
///
/// ORDERING CONSTRAINT: This function must be called AFTER the splitter has
/// committed its output via `apply_plan_to` (which bumps
/// `class_to_vreg.split_generation`). The splitter may truncate block-param
/// segments so they no longer cover `BLOCK_ENTRY`; calling this function
/// before the splitter runs would collect stale pre-split VRegs.
pub fn collect_block_param_vregs_per_block(
    func: &Function,
    egraph: &crate::egraph::EGraph,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    class_to_vreg: &ClassVRegMap,
) -> Vec<BTreeSet<VReg>> {
    debug_assert!(
        class_to_vreg.split_generation > 0,
        "collect_block_param_vregs_per_block called before splitter committed output \
         (class_to_vreg.split_generation == 0); call apply_plan_to first"
    );
    let n = func.blocks.len();
    let mut result: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); n];

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let entry_point = ProgramPoint::block_entry(block_idx);
        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block.id, pidx))
                && let Some(vreg) = class_to_vreg.lookup(egraph.find_immutable(cid), entry_point)
            {
                result[block_idx].insert(vreg);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn iconst_inst(dst: u32, val: i64) -> ScheduledInst {
        ScheduledInst {
            op: Op::Iconst(val, Type::I64),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Proj0,
            dst: VReg(dst),
            operands: vec![VReg(src)],
        }
    }

    fn add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::X86Add,
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    fn empty_phi_uses(n: usize) -> Vec<BTreeSet<VReg>> {
        vec![BTreeSet::new(); n]
    }

    // Test 1: Straight-line CFG (0 -> 1 -> 2).
    // Value defined in block 0, used only in block 2.
    // It should be live_out[0], live_in[1], live_out[1], live_in[2].
    #[test]
    fn straight_line_cross_block_liveness() {
        // Block 0: v0 = iconst
        // Block 1: v1 = iconst (v0 not used here, just passes through)
        // Block 2: v2 = use(v0)
        let schedules = vec![
            vec![iconst_inst(0, 1)], // block 0
            vec![iconst_inst(1, 2)], // block 1 (v0 passes through)
            vec![use_inst(2, 0)],    // block 2
        ];
        // 0 -> 1 -> 2
        let successors = vec![vec![1usize], vec![2], vec![]];
        let phi_uses = empty_phi_uses(3);

        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);

        // v0 defined in block 0, used in block 2.
        assert!(gl.live_out[0].contains(&VReg(0)), "v0 live_out of block 0");
        assert!(gl.live_in[1].contains(&VReg(0)), "v0 live_in of block 1");
        assert!(gl.live_out[1].contains(&VReg(0)), "v0 live_out of block 1");
        assert!(gl.live_in[2].contains(&VReg(0)), "v0 live_in of block 2");

        // v0 should NOT be live_out of block 2 (no successors use it).
        assert!(
            !gl.live_out[2].contains(&VReg(0)),
            "v0 not live_out of block 2"
        );
    }

    // Test 2: Diamond CFG (0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3).
    // Value defined in block 0, used in block 3.
    #[test]
    fn diamond_cfg_liveness() {
        // Block 0: v0 = iconst
        // Block 1: v1 = iconst (v0 passes through)
        // Block 2: v2 = iconst (v0 passes through)
        // Block 3: v3 = use(v0)
        let schedules = vec![
            vec![iconst_inst(0, 1)], // block 0
            vec![iconst_inst(1, 2)], // block 1
            vec![iconst_inst(2, 3)], // block 2
            vec![use_inst(3, 0)],    // block 3
        ];
        // 0 -> {1, 2}, 1 -> 3, 2 -> 3
        let successors = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let phi_uses = empty_phi_uses(4);

        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);

        assert!(gl.live_out[0].contains(&VReg(0)));
        assert!(gl.live_in[1].contains(&VReg(0)));
        assert!(gl.live_out[1].contains(&VReg(0)));
        assert!(gl.live_in[2].contains(&VReg(0)));
        assert!(gl.live_out[2].contains(&VReg(0)));
        assert!(gl.live_in[3].contains(&VReg(0)));
    }

    // Test 3: Loop CFG (0 -> 1 -> 0, with loop).
    // Value defined in block 0, used in block 1.
    #[test]
    fn loop_cfg_liveness_converges() {
        // Block 0: v0 = iconst
        // Block 1: v1 = use(v0) -- then jumps back to block 0
        let schedules = vec![
            vec![iconst_inst(0, 1)], // block 0
            vec![use_inst(1, 0)],    // block 1
        ];
        // 0 -> 1, 1 -> 0 (back-edge)
        let successors = vec![vec![1], vec![0]];
        let phi_uses = empty_phi_uses(2);

        // Should not infinite-loop and should converge.
        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);

        // v0 is defined in block 0 and used in block 1.
        assert!(gl.live_out[0].contains(&VReg(0)));
        assert!(gl.live_in[1].contains(&VReg(0)));
    }

    // Test 4: Value defined and used only within one block.
    // Should not appear in any block's live_in or live_out.
    #[test]
    fn block_local_value_not_in_live_sets() {
        // Block 0: v0 = iconst; v1 = add(v0, v0) -- both local
        // Block 1: v2 = iconst
        let schedules = vec![
            vec![iconst_inst(0, 1), add_inst(1, 0, 0)], // block 0
            vec![iconst_inst(2, 2)],                    // block 1
        ];
        let successors = vec![vec![1], vec![]];
        let phi_uses = empty_phi_uses(2);

        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);

        // v0 and v1 are local to block 0 -- should not be live across any boundary.
        assert!(!gl.live_out[0].contains(&VReg(0)));
        assert!(!gl.live_in[1].contains(&VReg(0)));
        assert!(!gl.live_out[0].contains(&VReg(1)));
    }

    // Test 5: phi_uses propagation.
    // A value used only in Jump args (terminator) of block 0 must be
    // upward-exposed from block 0.
    #[test]
    fn phi_uses_propagated() {
        // Block 0: v0 = iconst, then jumps with v0 as phi arg (in phi_uses[0])
        // Block 1: v1 = iconst
        let schedules = vec![
            vec![iconst_inst(0, 1)], // block 0
            vec![iconst_inst(1, 2)], // block 1
        ];
        let successors = vec![vec![1], vec![]];
        // v0 is used as a phi source at the terminator of block 0.
        let mut phi_uses = empty_phi_uses(2);
        phi_uses[0].insert(VReg(0));

        let gl = compute_global_liveness(&schedules, &successors, &phi_uses);

        // v0 is defined in block 0 but also used in phi_uses[0], so it stays local.
        // It should NOT be live_out[0] unless a successor needs it.
        // Actually phi_uses contribute to use(B), but if defined in B it won't be
        // upward-exposed. The live_out depends on successors' live_in.
        // Block 1 doesn't use v0, so it's not in live_in[1], so live_out[0] won't have it.
        // This is correct: phi_uses track what the terminator consumes locally.
        assert!(
            !gl.live_in[0].contains(&VReg(0)),
            "v0 is defined in block 0, not upward-exposed"
        );
    }
}
