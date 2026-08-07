use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::{MachOp, Op, PseudoOp};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::RegClass;

use super::liveness::LivenessInfo;

/// A set of VReg indices held as a sorted `Vec`.
///
/// The interference graph is the allocator's densest structure -- every program
/// point contributes an all-pairs edge insert over its live set, and coalescing
/// unions whole neighbourhoods -- so the per-element cost of the set is what the
/// pass costs. A `BTreeSet` pays an allocation and a tree descent per insert; a
/// sorted `Vec` pays a binary search and a memmove over a contiguous run, and
/// stores four bytes per neighbour instead of a node.
///
/// `LivenessInfo` holds its per-program-point sets in the same form, for the
/// same reason: it copies the running live set once per instruction. So does
/// `GlobalLiveness`, whose fixpoint rebuilds a block's live-out from every
/// successor's live-in on every round; there the whole update is one
/// [`VRegSet::union_minus`] merge per edge rather than an insert per member.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct VRegSet {
    elems: Vec<u32>,
}

impl VRegSet {
    pub fn new() -> Self {
        Self { elems: Vec::new() }
    }

    /// Insert `v`; returns whether it was absent.
    pub fn insert(&mut self, v: usize) -> bool {
        match self.elems.binary_search(&(v as u32)) {
            Ok(_) => false,
            Err(i) => {
                self.elems.insert(i, v as u32);
                true
            }
        }
    }

    /// Remove `v`; returns whether it was present.
    pub fn remove(&mut self, v: usize) -> bool {
        match self.elems.binary_search(&(v as u32)) {
            Ok(i) => {
                self.elems.remove(i);
                true
            }
            Err(_) => false,
        }
    }

    pub fn contains(&self, v: usize) -> bool {
        self.elems.binary_search(&(v as u32)).is_ok()
    }

    pub fn len(&self) -> usize {
        self.elems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    pub fn clear(&mut self) {
        self.elems.clear();
    }

    /// Ascending, which is what every caller that cares about order wants.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.elems.iter().map(|&v| v as usize)
    }

    /// Add every member of `other` that is not a member of `minus`.
    ///
    /// The dataflow in [`super::global_liveness`] is written entirely in this
    /// shape -- `live_out(B) = union of live_in(S) - params(S)`, `live_in(B) =
    /// use(B) + (live_out(B) - def(B))` -- and doing it a member at a time costs
    /// a binary search and a memmove each. Both operands are sorted, so one
    /// merge pass builds the answer.
    pub fn union_minus(&mut self, other: &VRegSet, minus: &VRegSet) {
        let mut merged = Vec::with_capacity(self.elems.len() + other.elems.len());
        let (mut i, mut j, mut k) = (0, 0, 0);
        while j < other.elems.len() {
            let v = other.elems[j];
            // Advance the exclusion cursor to the first member >= v.
            while k < minus.elems.len() && minus.elems[k] < v {
                k += 1;
            }
            j += 1;
            if minus.elems.get(k) == Some(&v) {
                continue;
            }
            while i < self.elems.len() && self.elems[i] < v {
                merged.push(self.elems[i]);
                i += 1;
            }
            if self.elems.get(i) == Some(&v) {
                i += 1;
            }
            merged.push(v);
        }
        merged.extend_from_slice(&self.elems[i..]);
        self.elems = merged;
    }

    /// Add every member of `other`.
    pub fn union_with(&mut self, other: &VRegSet) {
        self.union_minus(other, &VRegSet::new());
    }
}

impl FromIterator<usize> for VRegSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut elems: Vec<u32> = iter.into_iter().map(|v| v as u32).collect();
        elems.sort_unstable();
        elems.dedup();
        Self { elems }
    }
}

impl<'a> IntoIterator for &'a VRegSet {
    type Item = usize;
    type IntoIter = std::iter::Map<std::slice::Iter<'a, u32>, fn(&u32) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.elems.iter().map(|&v| v as usize)
    }
}

pub struct InterferenceGraph {
    pub num_vregs: usize,
    /// Adjacency list: VReg index -> set of interfering VReg indices.
    pub adj: Vec<VRegSet>,
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

    // The def interferes with everything live at its program point, except the
    // one operand it is supposed to overwrite.
    //
    // x86 arithmetic is two-address: `add dst, src` reads and writes `dst`, so
    // the result and `operands[0]` of every op in `Op::two_address_src` are
    // meant to be one register, and `lower.rs` emits `mov dst, operands[0]`
    // when they are not. `live_at[i]` is the set live *before* instruction i,
    // which contains that operand, so the blanket rule gave the def an edge to
    // the very value it wants to reuse -- making the two-address form
    // unsatisfiable by construction and every such op cost a copy. On the
    // `bench` kernels that was 246 of 875 emitted register-to-register moves,
    // against `gcc -O2`'s 345 in total.
    //
    // The exception is exactly one operand and only when it dies here. Two
    // things do not qualify, and both would be miscompiles:
    //
    // - an operand still live after this point. It is not being consumed, so
    //   the register cannot be taken from it.
    // - any *other* operand, dying or not. Lowering writes `dst` with the copy
    //   from `operands[0]` before the instruction reads them, so a second
    //   operand sharing `dst` is read after it has been overwritten.
    for (i, inst) in insts.iter().enumerate() {
        if inst.op.has_no_result() {
            continue;
        }
        let dst_idx = inst.dst.0 as usize;
        if dst_idx >= graph.num_vregs {
            continue;
        }
        let live_after: &VRegSet = if i + 1 < liveness.live_at.len() {
            &liveness.live_at[i + 1]
        } else {
            &liveness.live_out
        };
        let reusable = inst
            .op
            .two_address_src()
            .and_then(|k| inst.operands.get(k))
            .filter(|v| !live_after.contains(v.0 as usize))
            .filter(|v| inst.operands.iter().filter(|o| o == v).count() == 1)
            .map(|v| v.0 as usize);
        let dst_class = graph.reg_class[dst_idx];
        for live_idx in &liveness.live_at[i] {
            if live_idx < graph.num_vregs
                && graph.reg_class[live_idx] == dst_class
                && live_idx != dst_idx
                && Some(live_idx) != reusable
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
pub fn dying_clobber_operands(inst: &ScheduledInst, live_after: &VRegSet) -> BTreeSet<usize> {
    let operands: &[VReg] = match inst.op {
        Op::Pseudo(PseudoOp::CallResult(_, _)) | Op::Pseudo(PseudoOp::VoidCallBarrier) => {
            &inst.operands
        }
        Op::Mach(MachOp::X86Idiv(..)) | Op::Mach(MachOp::X86Div(..)) => {
            &inst.operands[..inst.operands.len().min(1)]
        }
        _ => &[],
    };
    operands
        .iter()
        .filter(|v| !live_after.contains(v.0 as usize))
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
                || !graph.adj[va].contains(vb)
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
            for idx in live_set {
                if idx > max_idx {
                    max_idx = idx;
                }
            }
        }
        for idx in &liveness.live_in {
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
        adj: vec![VRegSet::new(); num_vregs],
        reg_class,
    };

    build_interference_into(&mut graph, liveness, insts);

    graph
}

fn add_interferences_in_set(graph: &mut InterferenceGraph, live_set: &VRegSet) {
    let live: Vec<usize> = live_set.iter().collect();
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

    fn set(elems: &[usize]) -> VRegSet {
        elems.iter().copied().collect()
    }

    #[test]
    fn union_minus_adds_what_the_exclusion_does_not_cover() {
        let mut a = set(&[1, 5, 9]);
        a.union_minus(&set(&[0, 5, 6, 7]), &set(&[6]));
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![0, 1, 5, 7, 9]);
    }

    #[test]
    fn union_minus_keeps_a_member_the_exclusion_names_but_the_addend_does_not() {
        // The exclusion applies to `other`, not to what `self` already holds:
        // in the dataflow, `def(B)` removes a value from what flows in from
        // below, and says nothing about `use(B)`.
        let mut a = set(&[3]);
        a.union_minus(&set(&[3, 4]), &set(&[3, 4]));
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn union_with_is_a_union() {
        let mut a = set(&[2, 4]);
        a.union_with(&set(&[1, 4, 8]));
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![1, 2, 4, 8]);
        let mut empty = VRegSet::new();
        empty.union_with(&set(&[7]));
        assert_eq!(empty.iter().collect::<Vec<_>>(), vec![7]);
    }
    use crate::ir::op::{Op, PureOp};
    use crate::ir::types::Type;
    use crate::regalloc::liveness::compute_liveness;
    use crate::schedule::scheduler::ScheduledInst;

    fn iconst_inst(dst: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::Iconst(dst as i64, Type::I64)),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Mach(MachOp::X86Add),
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
        let live_out = VRegSet::new();
        let liveness = compute_liveness(&insts, &live_out);
        let classes = default_classes(&insts);
        let graph = build_interference(&liveness, &insts, &classes);

        // v0 and v1 are both live before inst 2, so they should interfere.
        assert!(
            graph.adj[0].contains(1),
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
                op: Op::Pure(PureOp::Proj0),
                dst: VReg(1),
                operands: vec![VReg(0)],
            },
            iconst_inst(2),
            ScheduledInst {
                op: Op::Pure(PureOp::Proj0),
                dst: VReg(3),
                operands: vec![VReg(2)],
            },
        ];
        let live_out = VRegSet::new();
        let liveness = compute_liveness(&insts, &live_out);
        let classes = default_classes(&insts);
        let graph = build_interference(&liveness, &insts, &classes);

        // v0 is used at inst 1 and then dead; v2 is defined at inst 2.
        // v0 and v2 should not be simultaneously live.
        assert!(
            !graph.adj[0].contains(2),
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
        let live_out = VRegSet::new();
        let liveness = compute_liveness(&insts, &live_out);
        let mut classes = BTreeMap::new();
        classes.insert(VReg(0), RegClass::GPR);
        classes.insert(VReg(1), RegClass::XMM);
        classes.insert(VReg(2), RegClass::GPR);
        let graph = build_interference(&liveness, &insts, &classes);

        // v0 (GPR) and v1 (XMM) are simultaneously live but different classes.
        assert!(
            !graph.adj[0].contains(1),
            "GPR v0 and XMM v1 should NOT interfere"
        );
    }
}
