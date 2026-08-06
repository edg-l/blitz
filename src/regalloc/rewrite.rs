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
    CoalesceAliases::new(coalesced).apply(insts)
}

/// Where each coalesced VReg ends up, resolved once for the whole function.
///
/// The chain a merged VReg leads along is followed at construction and the
/// answer stored flat, so renaming an operand is one index rather than a walk of
/// map lookups -- and the table is built once rather than once per block, which
/// is what a per-block `apply_coalescing` was doing with the same input.
pub struct CoalesceAliases {
    /// `target[i]` is what VReg `i` becomes, or `i` itself.
    target: Vec<u32>,
}

impl CoalesceAliases {
    pub fn new(coalesced: &[(usize, usize)]) -> Self {
        let max = coalesced
            .iter()
            .map(|&(into, from)| into.max(from))
            .max()
            .map_or(0, |m| m + 1);
        let mut target: Vec<u32> = (0..max as u32).collect();
        for &(into, from) in coalesced {
            target[from] = into as u32;
        }
        // Collapse the chains. Every step moves to a VReg that is itself
        // resolved next, so one forward pass would not finish the job; each
        // entry is walked to its end and the whole path pointed there.
        let mut path: Vec<u32> = Vec::new();
        for i in 0..max {
            let mut cur = i as u32;
            path.clear();
            while target[cur as usize] != cur {
                path.push(cur);
                cur = target[cur as usize];
            }
            for &node in &path {
                target[node as usize] = cur;
            }
        }
        CoalesceAliases { target }
    }

    pub fn resolve(&self, v: VReg) -> VReg {
        match self.target.get(v.0 as usize) {
            Some(&t) => VReg(t),
            None => v,
        }
    }

    pub fn apply(&self, insts: &[ScheduledInst]) -> Vec<ScheduledInst> {
        if self.target.is_empty() {
            return insts.to_vec();
        }
        insts
            .iter()
            .map(|inst| ScheduledInst {
                op: inst.op.clone(),
                dst: self.resolve(inst.dst),
                operands: inst.operands.iter().map(|&op| self.resolve(op)).collect(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::{Op, PureOp};
    use crate::ir::types::Type;

    fn iconst_inst(dst: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::Iconst(dst as i64, Type::I64)),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::Proj0),
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
