use std::collections::BTreeMap;

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;

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
    let mut alias: BTreeMap<u32, u32> = BTreeMap::new();
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
