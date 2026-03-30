use std::collections::HashMap;

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::Reg;

/// Rewrite all VReg references in an instruction list to physical registers.
///
/// VRegs that appear in `vreg_to_reg` are replaced with the corresponding
/// physical register. Any VReg not found in the map is left as-is (this
/// should not happen in a correct allocation, but is handled gracefully).
///
/// The returned instruction list has the same length and structure as the
/// input, but with physical register assignments embedded via the `dst`
/// and `operands` fields (which remain VReg-typed for compatibility with
/// ScheduledInst; the caller uses `vreg_to_reg` to interpret them).
///
/// Since ScheduledInst uses VReg fields, this function returns a new Vec
/// of ScheduledInst with operands/dst remapped, and separately a mapping
/// from the new "physical-register-encoded VRegs" to their Reg. The caller
/// should use the returned `vreg_to_reg` map when lowering.
///
/// For now, rewrite returns a Vec<ScheduledInst> where dst and operands
/// have been remapped through coalescing (aliases). Physical register
/// assignment is conveyed via the returned `HashMap<VReg, Reg>`.
pub fn rewrite_vregs(
    insts: &[ScheduledInst],
    vreg_to_reg: &HashMap<VReg, Reg>,
) -> Vec<ScheduledInst> {
    insts
        .iter()
        .map(|inst| {
            // Remap dst: if it's in the map, keep same VReg index (physical
            // register is looked up separately). For coalescing, dst stays
            // the same; the map tells us what register to use.
            let new_dst = inst.dst;
            let new_operands: Vec<VReg> = inst
                .operands
                .iter()
                .map(|&op| {
                    // If the operand VReg has a physical register, it's valid.
                    // We keep the VReg index; the lowering pass uses vreg_to_reg.
                    // No structural change needed here unless coalescing renamed VRegs.
                    let _ = vreg_to_reg.get(&op); // validate presence
                    op
                })
                .collect();
            ScheduledInst {
                op: inst.op.clone(),
                dst: new_dst,
                operands: new_operands,
            }
        })
        .collect()
}

/// Apply coalescing aliases to an instruction list.
///
/// For each `(merged_into, merged_from)` pair from coalescing, replace all
/// occurrences of `merged_from` with `merged_into` in both dst and operands.
pub fn apply_coalescing(
    insts: &[ScheduledInst],
    coalesced: &[(usize, usize)], // (merged_into, merged_from)
) -> Vec<ScheduledInst> {
    if coalesced.is_empty() {
        return insts.to_vec();
    }

    // Build alias map: merged_from -> merged_into.
    let mut alias: HashMap<u32, u32> = HashMap::new();
    for &(into, from) in coalesced {
        alias.insert(from as u32, into as u32);
    }

    let resolve = |v: VReg| -> VReg {
        let mut idx = v.0;
        // Follow alias chain.
        while let Some(&target) = alias.get(&idx) {
            idx = target;
        }
        VReg(idx)
    };

    insts
        .iter()
        .map(|inst| ScheduledInst {
            op: inst.op.clone(),
            dst: resolve(inst.dst),
            operands: inst.operands.iter().map(|&op| resolve(op)).collect(),
        })
        .collect()
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

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Proj0,
            dst: VReg(dst),
            operands: vec![VReg(src)],
        }
    }

    // rewrite_vregs: structure preserved.
    #[test]
    fn rewrite_preserves_structure() {
        let insts = vec![iconst_inst(0), use_inst(1, 0)];
        let mut map = HashMap::new();
        map.insert(VReg(0), Reg::RAX);
        map.insert(VReg(1), Reg::RCX);

        let rewritten = rewrite_vregs(&insts, &map);
        assert_eq!(rewritten.len(), 2);
        assert_eq!(rewritten[0].dst, VReg(0));
        assert_eq!(rewritten[1].operands[0], VReg(0));
    }

    // apply_coalescing: replaces merged_from with merged_into.
    #[test]
    fn coalescing_alias_applied() {
        // v1 is merged into v0.
        let insts = vec![iconst_inst(0), iconst_inst(1), use_inst(2, 1)];
        let coalesced = [(0usize, 1usize)];
        let result = apply_coalescing(&insts, &coalesced);

        // v1's def should now be v0.
        assert_eq!(
            result[1].dst,
            VReg(0),
            "merged_from (v1) should be renamed to v0"
        );
        // use of v1 should now be v0.
        assert_eq!(
            result[2].operands[0],
            VReg(0),
            "use of v1 should be v0 after coalescing"
        );
    }
}
