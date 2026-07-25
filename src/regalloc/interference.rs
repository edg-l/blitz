use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::Op;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::RegClass;

use super::liveness::LivenessInfo;

pub struct InterferenceGraph {
    pub num_vregs: usize,
    /// Adjacency list: VReg index -> set of interfering VReg indices.
    pub adj: Vec<BTreeSet<usize>>,
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

/// Add interference edges into an existing graph from per-block liveness.
///
/// Assumes `graph.reg_class` is already fully populated for all VReg indices
/// that may appear in `liveness` or `insts`. Only adds edges; does not resize
/// the graph or overwrite `reg_class`.
///
/// This is the inner workhorse called by both `build_interference` (per-block
/// path, which allocates and populates `reg_class` itself) and the global
/// allocator path (which pre-populates `reg_class` from all blocks before
/// calling this for each block).
pub fn build_interference_into(
    graph: &mut InterferenceGraph,
    liveness: &LivenessInfo,
    insts: &[ScheduledInst],
) {
    // For each program point (live_at[i] = live before inst i):
    // all simultaneously live VRegs of the same register class interfere.
    for live_set in &liveness.live_at {
        add_interferences_in_set(graph, live_set);
    }

    // The def interferes with everything live at its program point.
    for (i, inst) in insts.iter().enumerate() {
        let dst_idx = inst.dst.0 as usize;
        if dst_idx >= graph.num_vregs {
            continue;
        }
        let dst_class = graph.reg_class[dst_idx];
        for &live_vreg in &liveness.live_at[i] {
            let live_idx = live_vreg.0 as usize;
            if live_idx < graph.num_vregs
                && graph.reg_class[live_idx] == dst_class
                && live_idx != dst_idx
            {
                graph.add_edge(dst_idx, live_idx);
            }
        }
    }
}

/// The operands a clobbering instruction consumes and that die at it, which
/// its clobber phantoms must therefore *not* interfere with.
///
/// A phantom says "this register is overwritten here, so nothing living across
/// this point may hold it". An operand the instruction reads and that dies
/// there is not living across it: it is supposed to be in the clobbered
/// register. Excluding it removes an edge that is not real, and a spurious edge
/// against a pre-colored operand is unresolvable -- the colorer honours both
/// pre-colorings and the conflict passes silently into the assignment.
///
/// Which operands qualify depends on the instruction:
///
/// - a call consumes every argument in its ABI register, so all dying operands
///   qualify;
/// - a division consumes only the dividend (operand 0) in RAX. The divisor is
///   *not* excluded even though it usually dies there too: `cqo` writes RDX and
///   `idiv` writes both RAX and RDX before reading the divisor is finished, so
///   a divisor in either register is destroyed mid-instruction.
///
/// `live_after` is the live set at the following program point.
pub fn dying_clobber_operands(
    inst: &ScheduledInst,
    live_after: &BTreeSet<VReg>,
) -> BTreeSet<usize> {
    let operands: &[VReg] = match inst.op {
        Op::CallResult(_, _) | Op::VoidCallBarrier => &inst.operands,
        Op::X86Idiv | Op::X86Div => &inst.operands[..inst.operands.len().min(1)],
        _ => &[],
    };
    operands
        .iter()
        .filter(|v| !live_after.contains(v))
        .map(|v| v.0 as usize)
        .collect()
}

/// Pick pre-colorings to drop so that no two that remain interfere while
/// sharing a color. Returns the dropped VReg indices; the caller does its own
/// bookkeeping for them.
///
/// Two *phantom* pre-colorings never collide -- a phantom takes edges only to
/// the values live across its point, never to another phantom -- and a phantom
/// against a real value is handled before this, by dropping the real one. What
/// is left is two real pre-colorings claiming one register over a range where
/// both are live, and the later definition destroys the earlier.
///
/// The reachable case is a call result. `add_call_precolors_for_block` pins the
/// first result of a call to RAX under a `call_count == 1` guard, which holds
/// for the block it examines but not once `allocate_global` aggregates every
/// block's pre-colors into one function-wide list: two results from two
/// single-call blocks then both claim RAX.
///
/// Dropping either side is safe, because every pre-coloring reaching here is a
/// preference the lowering can restore with a move -- `mov dst, rax` after a
/// call, `mov rcx, count` before a variable shift, the entry move for a
/// parameter. The parameter is kept where the choice arises, since its move is
/// the one that costs a prologue instruction.
pub fn resolve_precoloring_conflicts(
    pre_coloring: &BTreeMap<usize, u32>,
    graph: &InterferenceGraph,
    param_vreg_indices: &BTreeSet<usize>,
) -> Vec<usize> {
    let entries: Vec<(usize, u32)> = pre_coloring.iter().map(|(&v, &c)| (v, c)).collect();
    let mut dropped: BTreeSet<usize> = BTreeSet::new();

    for (i, &(va, ca)) in entries.iter().enumerate() {
        if dropped.contains(&va) || va >= graph.num_vregs {
            continue;
        }
        for &(vb, cb) in &entries[i + 1..] {
            if ca != cb
                || dropped.contains(&vb)
                || vb >= graph.num_vregs
                || !graph.adj[va].contains(&vb)
            {
                continue;
            }
            let a_is_param = param_vreg_indices.contains(&va);
            let b_is_param = param_vreg_indices.contains(&vb);
            let victim = match (a_is_param, b_is_param) {
                (true, false) => vb,
                (false, true) => va,
                _ => va.max(vb),
            };
            dropped.insert(victim);
            if victim == va {
                break;
            }
        }
    }

    dropped.into_iter().collect()
}

/// Build an interference graph from liveness information.
///
/// For each program point, all simultaneously live VRegs of the same register
/// class interfere with each other.
///
/// Additionally, a def always interferes with all other VRegs live at the same
/// point (to handle the case where a definition and live range overlap at the
/// same instruction boundary).
///
/// This is a thin wrapper around `build_interference_into` for the per-block
/// path. The per-block path allocates and populates `reg_class` here before
/// delegating to the inner function.
pub fn build_interference(
    liveness: &LivenessInfo,
    insts: &[ScheduledInst],
    vreg_classes: &BTreeMap<VReg, RegClass>,
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
        adj: vec![BTreeSet::new(); num_vregs],
        reg_class,
    };

    build_interference_into(&mut graph, liveness, insts);

    graph
}

fn add_interferences_in_set(graph: &mut InterferenceGraph, live_set: &BTreeSet<VReg>) {
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

    fn default_classes(insts: &[ScheduledInst]) -> BTreeMap<VReg, RegClass> {
        let mut m = BTreeMap::new();
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
        let live_out = BTreeSet::new();
        let liveness = compute_liveness(&insts, &live_out);
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
        let live_out = BTreeSet::new();
        let liveness = compute_liveness(&insts, &live_out);
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
        let live_out = BTreeSet::new();
        let liveness = compute_liveness(&insts, &live_out);
        let mut classes = BTreeMap::new();
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
