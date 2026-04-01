use std::collections::{HashMap, HashSet};

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::RegClass;

use super::liveness::LivenessInfo;

pub struct InterferenceGraph {
    pub num_vregs: usize,
    /// Adjacency list: VReg index -> set of interfering VReg indices.
    pub adj: Vec<HashSet<usize>>,
    /// Register class of each VReg.
    pub reg_class: Vec<RegClass>,
}

impl InterferenceGraph {
    /// Add an interference edge between two VRegs (undirected).
    /// No-op if they are the same or already adjacent.
    pub fn add_edge(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.adj[a].insert(b);
        self.adj[b].insert(a);
    }
}

/// Build an interference graph from liveness information.
///
/// For each program point, all simultaneously live VRegs of the same register
/// class interfere with each other.
///
/// Additionally, a def always interferes with all other VRegs live at the same
/// point (to handle the case where a definition and live range overlap at the
/// same instruction boundary).
pub fn build_interference(
    liveness: &LivenessInfo,
    insts: &[ScheduledInst],
    vreg_classes: &HashMap<VReg, RegClass>,
) -> InterferenceGraph {
    // Determine the total number of VRegs (max index + 1).
    let num_vregs = {
        let mut max_idx = 0usize;
        for inst in insts {
            let idx = inst.dst.0 as usize;
            if idx > max_idx {
                max_idx = idx;
            }
            for &op in &inst.operands {
                let oidx = op.0 as usize;
                if oidx > max_idx {
                    max_idx = oidx;
                }
            }
        }
        // Also check VRegs that appear in live_at (e.g. live-in from predecessors).
        for live_set in &liveness.live_at {
            for v in live_set {
                let idx = v.0 as usize;
                if idx > max_idx {
                    max_idx = idx;
                }
            }
        }
        for v in &liveness.live_in {
            let idx = v.0 as usize;
            if idx > max_idx {
                max_idx = idx;
            }
        }
        max_idx + 1
    };

    let default_class = RegClass::GPR;
    let mut reg_class = vec![default_class; num_vregs];
    for (&vreg, &class) in vreg_classes {
        let idx = vreg.0 as usize;
        if idx < num_vregs {
            reg_class[idx] = class;
        }
    }

    let mut graph = InterferenceGraph {
        num_vregs,
        adj: vec![HashSet::new(); num_vregs],
        reg_class,
    };

    // For each program point (live_at[i] = live before inst i):
    // all simultaneously live VRegs of the same class interfere.
    for live_set in &liveness.live_at {
        add_interferences_in_set(&mut graph, live_set);
    }

    // Also add interferences at the point where each VReg is defined:
    // the def interferes with everything live right after the def (live_at[i]
    // without the def itself, which represents the set live *before* the def
    // after our backward-pass computation — but since we removed dst before
    // adding uses, live_at[i] is "live before" meaning the dst is NOT in
    // live_at[i] for its own instruction). We handle this by adding interferences
    // between the def and whatever is live at that point.
    for (i, inst) in insts.iter().enumerate() {
        let dst_idx = inst.dst.0 as usize;
        let dst_class = graph.reg_class[dst_idx];
        for &live_vreg in &liveness.live_at[i] {
            let live_idx = live_vreg.0 as usize;
            if graph.reg_class[live_idx] == dst_class && live_idx != dst_idx {
                graph.add_edge(dst_idx, live_idx);
            }
        }
    }

    graph
}

fn add_interferences_in_set(graph: &mut InterferenceGraph, live_set: &HashSet<VReg>) {
    let live: Vec<usize> = live_set.iter().map(|v| v.0 as usize).collect();
    for i in 0..live.len() {
        for j in (i + 1)..live.len() {
            let a = live[i];
            let b = live[j];
            // Only add interference if same register class.
            if graph.reg_class[a] == graph.reg_class[b] {
                graph.add_edge(a, b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;
    use crate::regalloc::liveness::compute_liveness;
    use crate::schedule::scheduler::ScheduledInst;

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

    fn default_classes(insts: &[ScheduledInst]) -> HashMap<VReg, RegClass> {
        let mut m = HashMap::new();
        for inst in insts {
            m.insert(inst.dst, RegClass::GPR);
            for &op in &inst.operands {
                m.insert(op, RegClass::GPR);
            }
        }
        m
    }

    // Overlapping ranges interfere.
    #[test]
    fn overlapping_ranges_interfere() {
        // v0 = iconst  (inst 0)
        // v1 = iconst  (inst 1)
        // v2 = add(v0, v1)  (inst 2) -- v0 and v1 live simultaneously
        let insts = vec![iconst_inst(0), iconst_inst(1), add_inst(2, 0, 1)];
        let live_out = HashSet::new();
        let liveness = compute_liveness(&insts, &live_out, &HashMap::new());
        let classes = default_classes(&insts);
        let graph = build_interference(&liveness, &insts, &classes);

        // v0 and v1 are both live before inst 2, so they should interfere.
        assert!(
            graph.adj[0].contains(&1),
            "v0 and v1 should interfere (both live before add)"
        );
    }

    // Non-overlapping ranges don't interfere.
    #[test]
    fn non_overlapping_no_interference() {
        // v0 = iconst  (inst 0) -- only used at inst 1
        // v1 = proj0(v0)  (inst 1) -- v0 dies here
        // v2 = iconst  (inst 2) -- v1 not used after inst 3
        // v3 = proj0(v2)  (inst 3)
        let insts = vec![
            iconst_inst(0),
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(1),
                operands: vec![VReg(0)],
            },
            iconst_inst(2),
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(3),
                operands: vec![VReg(2)],
            },
        ];
        let live_out = HashSet::new();
        let liveness = compute_liveness(&insts, &live_out, &HashMap::new());
        let classes = default_classes(&insts);
        let graph = build_interference(&liveness, &insts, &classes);

        // v0 is used at inst 1 and then dead; v2 is defined at inst 2.
        // v0 and v2 should not be simultaneously live.
        assert!(
            !graph.adj[0].contains(&2),
            "v0 and v2 should NOT interfere (non-overlapping live ranges)"
        );
    }

    // Cross-class: GPR and XMM don't interfere even with overlapping ranges.
    #[test]
    fn cross_class_no_interference() {
        let insts = vec![
            iconst_inst(0), // v0 = GPR
            iconst_inst(1), // v1 = XMM
            add_inst(2, 0, 1),
        ];
        let live_out = HashSet::new();
        let liveness = compute_liveness(&insts, &live_out, &HashMap::new());
        let mut classes = HashMap::new();
        classes.insert(VReg(0), RegClass::GPR);
        classes.insert(VReg(1), RegClass::XMM);
        classes.insert(VReg(2), RegClass::GPR);
        let graph = build_interference(&liveness, &insts, &classes);

        // v0 (GPR) and v1 (XMM) are simultaneously live but different classes.
        assert!(
            !graph.adj[0].contains(&1),
            "GPR v0 and XMM v1 should NOT interfere"
        );
    }
}
