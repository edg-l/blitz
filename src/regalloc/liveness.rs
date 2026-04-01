use std::collections::{HashMap, HashSet};

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;

pub struct LivenessInfo {
    /// For each program point (instruction index), the set of live VRegs.
    /// live_at[i] is the set live *before* instruction i executes.
    pub live_at: Vec<HashSet<VReg>>,
    /// Live-in set for the block (live before the first instruction).
    pub live_in: HashSet<VReg>,
    /// Live-out set for the block (live after the last instruction).
    pub live_out: HashSet<VReg>,
}

/// Compute liveness for a single basic block's scheduled instructions.
///
/// `block_live_out` is the set of VRegs live out of this block (used by
/// successors / block params).
///
/// Backward pass:
///   live = block_live_out
///   For each inst in reverse:
///     live_at[i] = live after removing dst, then adding uses
///     Remove dst from live (if this inst defines it)
///     Add all operands to live
///   live_in = live after processing all instructions
pub fn compute_liveness(
    insts: &[ScheduledInst],
    block_live_out: &HashSet<VReg>,
    deadlines: &HashMap<VReg, usize>,
) -> LivenessInfo {
    let n = insts.len();
    let mut live_at: Vec<HashSet<VReg>> = vec![HashSet::new(); n];
    let mut live: HashSet<VReg> = block_live_out.clone();

    // Pre-compute per-position deadline VRegs so we can insert them during the
    // backward pass without scanning the map each iteration.
    let mut deadline_at: Vec<Vec<VReg>> = vec![vec![]; n];
    for (&vreg, &pos) in deadlines {
        if pos < n {
            deadline_at[pos].push(vreg);
        }
        // If pos >= n, the VReg should already be in block_live_out.
    }

    for i in (0..n).rev() {
        // VRegs with deadline at position i enter live here.
        for &v in &deadline_at[i] {
            live.insert(v);
        }

        let inst = &insts[i];

        // Remove the definition: if this VReg is defined here, it's not live
        // before this instruction (in SSA form, each VReg is defined once).
        live.remove(&inst.dst);

        // Add uses: VRegs used by this instruction are live before it.
        for &op in &inst.operands {
            live.insert(op);
        }

        // live_at[i] = set of VRegs live before instruction i.
        live_at[i] = live.clone();
    }

    let live_in = live.clone();

    LivenessInfo {
        live_at,
        live_in,
        live_out: block_live_out.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn iconst_inst(dst: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Iconst(dst as i64, Type::I64),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::X86Add,
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Proj0,
            dst: VReg(dst),
            operands: vec![VReg(src)],
        }
    }

    // 10.2: Straight-line: VReg live from def to last use.
    //
    // v0 = iconst  (inst 0)
    // v1 = iconst  (inst 1)
    // v2 = add(v0, v1)  (inst 2)
    // v3 = use(v2)  (inst 3)
    //
    // v0 is live at instructions 0 (as result), 2 (as operand).
    // After backward pass: live_at[i] = live *before* inst i.
    // live_at[2] must contain v0 and v1 (used there).
    // live_at[0] must NOT contain v0 (it's defined there, not live before).
    #[test]
    fn straight_line_liveness() {
        let insts = vec![
            iconst_inst(0),    // v0 = iconst
            iconst_inst(1),    // v1 = iconst
            add_inst(2, 0, 1), // v2 = add(v0, v1)
            use_inst(3, 2),    // v3 = use(v2)
        ];
        let live_out: HashSet<VReg> = HashSet::new();
        let info = compute_liveness(&insts, &live_out, &HashMap::new());

        // Before inst 2 (add): v0 and v1 must be live.
        assert!(info.live_at[2].contains(&VReg(0)), "v0 live before inst 2");
        assert!(info.live_at[2].contains(&VReg(1)), "v1 live before inst 2");

        // Before inst 3 (use): v2 must be live.
        assert!(info.live_at[3].contains(&VReg(2)), "v2 live before inst 3");

        // v0 is not live before inst 0 (it's defined there).
        assert!(
            !info.live_at[0].contains(&VReg(0)),
            "v0 not live before its def"
        );

        // live_in: nothing is live before the block (all defs are inside).
        assert!(
            info.live_in.is_empty(),
            "no VRegs live in for this block: {:?}",
            info.live_in
        );
    }

    // 10.2: Cross-block: VReg in live_out propagated correctly.
    //
    // If v0 is in live_out (used by a successor), it should be live throughout.
    #[test]
    fn cross_block_live_out() {
        let insts = vec![
            iconst_inst(0), // v0 = iconst
            iconst_inst(1), // v1 = iconst
        ];
        let mut live_out: HashSet<VReg> = HashSet::new();
        live_out.insert(VReg(0)); // v0 is used in a successor block

        let info = compute_liveness(&insts, &live_out, &HashMap::new());

        // v0 should be in live_out.
        assert!(info.live_out.contains(&VReg(0)));

        // v0 is defined at inst 0, so before inst 0 it's not live.
        // But after inst 0 it should be live (since it's in live_out).
        // live_at[1] = live before inst 1: v0 should be here since live_out includes it
        // and nothing kills it after inst 0.
        assert!(
            info.live_at[1].contains(&VReg(0)),
            "v0 should be live at inst 1 since it's in live_out"
        );
    }

    // 10.2: Block parameter liveness.
    //
    // A VReg that appears only in live_out (used as a block param) should be
    // live_in if it's defined outside this block.
    #[test]
    fn block_param_liveness() {
        // The block uses v5 which is defined outside (in a predecessor).
        // Inst: v6 = use(v5)
        let insts = vec![use_inst(6, 5)];
        let live_out: HashSet<VReg> = HashSet::new();

        let info = compute_liveness(&insts, &live_out, &HashMap::new());

        // v5 is used in inst 0, so it must be live_in.
        assert!(
            info.live_in.contains(&VReg(5)),
            "v5 (external param) must be in live_in"
        );

        // live_at[0] must contain v5.
        assert!(
            info.live_at[0].contains(&VReg(5)),
            "v5 must be live before inst 0"
        );
    }
}
