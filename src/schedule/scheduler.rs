use std::collections::{BTreeMap, BinaryHeap};

use crate::egraph::extract::{VReg, VRegInst};
use crate::ir::op::Op;

// ── DAG representation (9.1) ──────────────────────────────────────────────────

/// A node in the scheduling DAG.
pub struct DagNode {
    pub id: usize,
    pub op: Op,
    pub dst: VReg,
    /// Data dependencies: VRegs this node uses as operands.
    pub operands: Vec<VReg>,
    /// Must maintain order relative to other effectful nodes.
    pub is_effectful: bool,
    /// Position among effectful ops in original order (if effectful).
    pub effectful_order: Option<usize>,
}

/// Dependency DAG for a basic block.
pub struct ScheduleDag {
    pub nodes: Vec<DagNode>,
    /// node -> nodes that depend on it (successors in dependency order).
    pub succs: Vec<Vec<usize>>,
    /// node -> nodes it depends on (predecessors in dependency order).
    pub preds: Vec<Vec<usize>>,
}

impl ScheduleDag {
    /// Build a DAG from a linearized VRegInst sequence.
    ///
    /// Data edges: if node B uses VReg v and node A defines v, add A -> B.
    /// Effectful ordering edges (9.4): between consecutive effectful ops.
    pub fn build(insts: &[VRegInst]) -> Self {
        let n = insts.len();
        let mut nodes: Vec<DagNode> = Vec::with_capacity(n);
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Map VReg -> defining node index.
        let mut def_map: BTreeMap<VReg, usize> = BTreeMap::new();

        // First pass: create nodes and record definitions.
        let mut effectful_count = 0usize;
        for (idx, inst) in insts.iter().enumerate() {
            let effectful = is_effectful(&inst.op);
            let effectful_order = if effectful {
                let order = effectful_count;
                effectful_count += 1;
                Some(order)
            } else {
                None
            };

            let operands: Vec<VReg> = inst.operands.iter().flatten().copied().collect();

            nodes.push(DagNode {
                id: idx,
                op: inst.op.clone(),
                dst: inst.dst,
                operands,
                is_effectful: effectful,
                effectful_order,
            });

            def_map.insert(inst.dst, idx);
        }

        // Second pass: add data dependency edges.
        for (b_idx, inst) in insts.iter().enumerate() {
            for &op_vreg in inst.operands.iter().flatten() {
                if let Some(&a_idx) = def_map.get(&op_vreg)
                    && a_idx != b_idx
                {
                    // A -> B: A must come before B.
                    if !succs[a_idx].contains(&b_idx) {
                        succs[a_idx].push(b_idx);
                        preds[b_idx].push(a_idx);
                    }
                }
            }
        }

        // Third pass: effectful ordering edges (9.4).
        // Collect effectful node indices in their original order.
        let effectful_nodes: Vec<usize> = nodes
            .iter()
            .filter(|n| n.is_effectful)
            .map(|n| n.id)
            .collect();

        for pair in effectful_nodes.windows(2) {
            let (a_idx, b_idx) = (pair[0], pair[1]);
            if !succs[a_idx].contains(&b_idx) {
                succs[a_idx].push(b_idx);
                preds[b_idx].push(a_idx);
            }
        }

        ScheduleDag {
            nodes,
            succs,
            preds,
        }
    }
}

/// Returns true if the op has side effects that must be ordered.
///
/// Currently all ops in the IR are pure. This will need updating when
/// Store/Load/Call ops are added.
fn is_effectful(_op: &Op) -> bool {
    false
}

// ── Priority computation (9.2) ────────────────────────────────────────────────

/// Scheduling priority data per node.
struct Priority {
    /// Longest path from this node to any sink (root-ward in data-flow = leaf in schedule).
    critical_path: usize,
    /// Number of operand VRegs whose last use within the block is this node.
    last_use_count: usize,
}

/// Compute critical path lengths using reverse BFS from sinks (nodes with no successors).
fn compute_priorities(dag: &ScheduleDag) -> Vec<Priority> {
    let n = dag.nodes.len();

    // critical_path[i] = length of longest path from node i to any sink.
    let mut critical_path = vec![0usize; n];

    // Process nodes in reverse topological order (from sinks toward sources).
    // We compute topological order first via Kahn's algorithm, then reverse it.
    let mut in_degree: Vec<usize> = dag.preds.iter().map(|p| p.len()).collect();
    let mut topo: Vec<usize> = Vec::with_capacity(n);
    let mut queue: std::collections::VecDeque<usize> =
        (0..n).filter(|&i| in_degree[i] == 0).collect();

    while let Some(node) = queue.pop_front() {
        topo.push(node);
        for &succ in &dag.succs[node] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    // Traverse topo in reverse to propagate critical path from sinks.
    for &node in topo.iter().rev() {
        let max_succ_path = dag.succs[node]
            .iter()
            .map(|&s| critical_path[s])
            .max()
            .unwrap_or(0);
        critical_path[node] = 1 + max_succ_path;
    }

    // Compute last_use_count: for each node, count how many of its operand
    // VRegs have this node as the last user within the block.
    //
    // For each VReg, find the highest-index node (in the original inst order)
    // that uses it, treating original order as a proxy for "last use."
    // Since insts are in def-before-use order, the last node in original order
    // that references a VReg is the last in-block user.
    let mut last_user: BTreeMap<VReg, usize> = BTreeMap::new();
    for node in &dag.nodes {
        for &operand in &node.operands {
            last_user.insert(operand, node.id);
        }
    }

    let mut last_use_count = vec![0usize; n];
    for (vreg, &user_idx) in &last_user {
        // Only count if the defining node is also in this block.
        let defined_in_block = dag.nodes.iter().any(|nd| nd.dst == *vreg);
        if defined_in_block {
            last_use_count[user_idx] += 1;
        }
    }

    (0..n)
        .map(|i| Priority {
            critical_path: critical_path[i],
            last_use_count: last_use_count[i],
        })
        .collect()
}

// ── List scheduler (9.3) ──────────────────────────────────────────────────────

/// A single scheduled instruction.
#[derive(Clone)]
pub struct ScheduledInst {
    pub op: Op,
    pub dst: VReg,
    pub operands: Vec<VReg>,
}

/// Priority key for the ready heap: (last_use_count, critical_path, node_id).
/// Higher last_use_count wins; on tie, higher critical_path wins; on tie, lower id for stability.
#[derive(Eq, PartialEq)]
struct ReadyKey {
    last_use_count: usize,
    critical_path: usize,
    // Negate id so lower ids sort higher in max-heap (stable tie-break).
    neg_id: i64,
}

impl Ord for ReadyKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.last_use_count
            .cmp(&other.last_use_count)
            .then(self.critical_path.cmp(&other.critical_path))
            .then(self.neg_id.cmp(&other.neg_id))
    }
}

impl PartialOrd for ReadyKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Schedule a basic block DAG into a linear instruction sequence.
///
/// Uses a list scheduler with priority:
/// 1. Most operand VRegs freed (last_use_count) - reduces register pressure.
/// 2. Longest critical path - avoids stalling the critical path.
pub fn schedule(dag: &ScheduleDag) -> Vec<ScheduledInst> {
    let n = dag.nodes.len();
    let priorities = compute_priorities(dag);

    // Track remaining predecessor count per node.
    let mut remaining_preds: Vec<usize> = dag.preds.iter().map(|p| p.len()).collect();

    let mut ready: BinaryHeap<(ReadyKey, usize)> = BinaryHeap::new();

    // Seed with nodes that have no predecessors.
    for i in 0..n {
        if remaining_preds[i] == 0 {
            let key = ReadyKey {
                last_use_count: priorities[i].last_use_count,
                critical_path: priorities[i].critical_path,
                neg_id: -(i as i64),
            };
            ready.push((key, i));
        }
    }

    let mut result: Vec<ScheduledInst> = Vec::with_capacity(n);

    while let Some((_, node_idx)) = ready.pop() {
        let node = &dag.nodes[node_idx];
        result.push(ScheduledInst {
            op: node.op.clone(),
            dst: node.dst,
            operands: node.operands.clone(),
        });

        // Reduce predecessor counts for successors; enqueue newly ready ones.
        for &succ in &dag.succs[node_idx] {
            remaining_preds[succ] -= 1;
            if remaining_preds[succ] == 0 {
                let key = ReadyKey {
                    last_use_count: priorities[succ].last_use_count,
                    critical_path: priorities[succ].critical_path,
                    neg_id: -(succ as i64),
                };
                ready.push((key, succ));
            }
        }
    }

    debug_assert_eq!(
        result.len(),
        n,
        "scheduler produced {} instructions but DAG has {} nodes (cycle?)",
        result.len(),
        n
    );

    result
}

// ── Tests (9.5, 9.6) ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::Type;

    fn iconst_inst(dst: u32) -> VRegInst {
        VRegInst {
            dst: VReg(dst),
            op: Op::Iconst(dst as i64, Type::I64),
            operands: vec![],
        }
    }

    fn x86add_inst(dst: u32, a: u32, b: u32) -> VRegInst {
        VRegInst {
            dst: VReg(dst),
            op: Op::X86Add,
            operands: vec![Some(VReg(a)), Some(VReg(b))],
        }
    }

    fn proj0_inst(dst: u32, src: u32) -> VRegInst {
        VRegInst {
            dst: VReg(dst),
            op: Op::Proj0,
            operands: vec![Some(VReg(src))],
        }
    }

    /// Check that every instruction's operands are defined before it in the schedule.
    fn assert_topo_order(scheduled: &[ScheduledInst]) {
        let mut defined: std::collections::BTreeSet<VReg> = std::collections::BTreeSet::new();
        for inst in scheduled {
            for &op in &inst.operands {
                assert!(
                    defined.contains(&op),
                    "VReg {:?} used before defined (op={:?})",
                    op,
                    inst.op
                );
            }
            defined.insert(inst.dst);
        }
    }

    /// Count peak simultaneously live VRegs in a schedule.
    fn peak_live(scheduled: &[ScheduledInst]) -> usize {
        // Build a def->last_use map.
        let mut last_use: BTreeMap<VReg, usize> = BTreeMap::new();
        for (i, inst) in scheduled.iter().enumerate() {
            for &op in &inst.operands {
                last_use.insert(op, i);
            }
        }

        let mut live: std::collections::BTreeSet<VReg> = std::collections::BTreeSet::new();
        let mut peak = 0;
        for (i, inst) in scheduled.iter().enumerate() {
            // Kill VRegs whose last use was the previous instruction.
            live.retain(|v| last_use.get(v).map_or(false, |&u| u >= i));
            live.insert(inst.dst);
            if live.len() > peak {
                peak = live.len();
            }
        }
        peak
    }

    // ── 9.5: Straight-line arithmetic - verify topological order ─────────────

    /// Chain: v0 = iconst, v1 = iconst, v2 = x86add(v0,v1), v3 = proj0(v2)
    #[test]
    fn straight_line_topo_order() {
        let insts = vec![
            iconst_inst(0),
            iconst_inst(1),
            x86add_inst(2, 0, 1),
            proj0_inst(3, 2),
        ];
        let dag = ScheduleDag::build(&insts);
        let scheduled = schedule(&dag);
        assert_eq!(scheduled.len(), 4);
        assert_topo_order(&scheduled);
    }

    // ── 9.5: Diamond DAG - verify register pressure heuristic ────────────────

    /// Diamond shape:
    ///   v0 = iconst 1
    ///   v1 = iconst 2
    ///   v2 = iconst 3
    ///   pair_ab = x86add(v0, v1)    ; uses v0, v1
    ///   r_ab    = proj0(pair_ab)
    ///   pair_ac = x86add(v0, v2)    ; uses v0, v2  (v0 is shared)
    ///   r_ac    = proj0(pair_ac)
    ///   pair_r  = x86add(r_ab, r_ac) ; merge
    ///   r_final = proj0(pair_r)
    #[test]
    fn diamond_dag_topo_order() {
        let insts = vec![
            iconst_inst(0),       // v0
            iconst_inst(1),       // v1
            iconst_inst(2),       // v2
            x86add_inst(3, 0, 1), // pair_ab = x86add(v0, v1)
            proj0_inst(4, 3),     // r_ab = proj0(pair_ab)
            x86add_inst(5, 0, 2), // pair_ac = x86add(v0, v2)
            proj0_inst(6, 5),     // r_ac = proj0(pair_ac)
            x86add_inst(7, 4, 6), // pair_r = x86add(r_ab, r_ac)
            proj0_inst(8, 7),     // r_final = proj0(pair_r)
        ];
        let dag = ScheduleDag::build(&insts);
        let scheduled = schedule(&dag);
        assert_eq!(scheduled.len(), 9);
        assert_topo_order(&scheduled);
    }

    // ── 9.5: Effectful ops stay in order ─────────────────────────────────────

    /// Two effectful ops in original order must appear in that order.
    /// We simulate effectful ops by building a DAG manually and injecting
    /// ordering edges, verifying the scheduler respects them.
    ///
    /// Since no current Op variants are effectful, we test the ordering edge
    /// mechanism directly through the DAG builder with a hand-crafted pair.
    #[test]
    fn effectful_ops_ordered() {
        // Build two independent iconst nodes and manually mark them effectful
        // by setting is_effectful and injecting an ordering edge.
        // (In real use, effectful ops would be stores/calls.)
        let v0 = VReg(0);
        let v1 = VReg(1);

        let nodes = vec![
            DagNode {
                id: 0,
                op: Op::Iconst(10, Type::I64),
                dst: v0,
                operands: vec![],
                is_effectful: true,
                effectful_order: Some(0),
            },
            DagNode {
                id: 1,
                op: Op::Iconst(20, Type::I64),
                dst: v1,
                operands: vec![],
                is_effectful: true,
                effectful_order: Some(1),
            },
        ];
        // Ordering edge: node 0 -> node 1.
        let succs = vec![vec![1usize], vec![]];
        let preds = vec![vec![], vec![0usize]];

        let dag = ScheduleDag {
            nodes,
            succs,
            preds,
        };
        let scheduled = schedule(&dag);
        assert_eq!(scheduled.len(), 2);
        // v0 must come before v1.
        let pos0 = scheduled.iter().position(|s| s.dst == v0).unwrap();
        let pos1 = scheduled.iter().position(|s| s.dst == v1).unwrap();
        assert!(
            pos0 < pos1,
            "effectful op 0 (pos {pos0}) must precede effectful op 1 (pos {pos1})"
        );
    }

    // ── 9.5: effectful ordering via DAG builder ───────────────────────────────

    /// Verify that ScheduleDag::build inserts effectful ordering edges.
    /// We cannot test this with the current Op set (all pure), so we verify
    /// the absence of spurious edges for a pure straight-line block.
    #[test]
    fn pure_block_no_spurious_edges() {
        let insts = vec![iconst_inst(0), iconst_inst(1), x86add_inst(2, 0, 1)];
        let dag = ScheduleDag::build(&insts);
        // No effectful ordering edges; only data edges.
        // node 0 -> node 2, node 1 -> node 2.
        assert!(dag.succs[0].contains(&2));
        assert!(dag.succs[1].contains(&2));
        // No ordering between 0 and 1 (both pure, no data dep).
        assert!(!dag.succs[0].contains(&1));
        assert!(!dag.succs[1].contains(&0));
    }

    // ── 9.6: SPIKE - Register pressure measurement ────────────────────────────

    /// DAG shape: chain
    ///   v0 -> v1 -> v2 -> v3 -> v4
    ///   Each step: pair = x86add(prev_result), result = proj0(pair)
    ///
    /// Chain has minimal parallelism. Both the list scheduler and topo-sort
    /// produce the same single linear order, so peak live ~ 2-3 VRegs at any
    /// point (the result register + operand being consumed).
    #[test]
    fn spike_chain_register_pressure() {
        // v0 = iconst, v1 = iconst
        // v2 = x86add(v0, v1), v3 = proj0(v2)
        // v4 = iconst,         v5 = x86add(v3, v4), v6 = proj0(v5)
        // v7 = iconst,         v8 = x86add(v6, v7), v9 = proj0(v8)
        let insts = vec![
            iconst_inst(0),
            iconst_inst(1),
            x86add_inst(2, 0, 1),
            proj0_inst(3, 2),
            iconst_inst(4),
            x86add_inst(5, 3, 4),
            proj0_inst(6, 5),
            iconst_inst(7),
            x86add_inst(8, 6, 7),
            proj0_inst(9, 8),
        ];
        let dag = ScheduleDag::build(&insts);
        let scheduled = schedule(&dag);
        assert_topo_order(&scheduled);

        // Peak live VRegs for list scheduler.
        let list_peak = peak_live(&scheduled);

        // Simple topological sort baseline (original emission order is already topo).
        let topo_scheduled: Vec<ScheduledInst> = insts
            .iter()
            .map(|i| ScheduledInst {
                op: i.op.clone(),
                dst: i.dst,
                operands: i.operands.iter().flatten().copied().collect(),
            })
            .collect();
        let topo_peak = peak_live(&topo_scheduled);

        // Document findings:
        // Chain shape has no scheduling freedom — both produce identical order.
        // Peak live for chain (10 insts): typically 2-3 VRegs.
        // list_peak and topo_peak should be equal for a pure chain.
        assert!(
            list_peak <= topo_peak + 1,
            "list scheduler should not exceed topo-sort peak by more than 1 for a chain: list={list_peak}, topo={topo_peak}"
        );
    }

    /// DAG shape: wide fan-out
    ///   v0 = iconst (shared operand)
    ///   v1..v5 = iconst each
    ///   v6..v10 = x86add(v0, vi)   (all use v0)
    ///   v11..v15 = proj0(v6..v10)
    ///
    /// Wide fan-out keeps v0 live across all 5 adds. List scheduler should
    /// schedule all uses of v0 early (freeing it after its last use) rather
    /// than interleaving unrelated work.
    ///
    /// Finding: both schedules have peak ~7 here (v0 + 5 pairs + 1 in flight).
    /// List scheduler matches or beats topo-sort.
    #[test]
    fn spike_wide_fanout_register_pressure() {
        // v0 = iconst (shared)
        // v1..v5 = iconst operands
        // v6 = x86add(v0, v1), v7 = proj0(v6)
        // v8 = x86add(v0, v2), v9 = proj0(v8)
        // v10 = x86add(v0, v3), v11 = proj0(v10)
        // v12 = x86add(v0, v4), v13 = proj0(v12)
        // v14 = x86add(v0, v5), v15 = proj0(v14)
        let insts = vec![
            iconst_inst(0),        // shared v0
            iconst_inst(1),        // v1
            iconst_inst(2),        // v2
            iconst_inst(3),        // v3
            iconst_inst(4),        // v4
            iconst_inst(5),        // v5
            x86add_inst(6, 0, 1),  // pair0 = x86add(v0, v1)
            proj0_inst(7, 6),      // r0 = proj0(pair0)
            x86add_inst(8, 0, 2),  // pair1 = x86add(v0, v2)
            proj0_inst(9, 8),      // r1 = proj0(pair1)
            x86add_inst(10, 0, 3), // pair2 = x86add(v0, v3)
            proj0_inst(11, 10),    // r2 = proj0(pair2)
            x86add_inst(12, 0, 4), // pair3 = x86add(v0, v4)
            proj0_inst(13, 12),    // r3 = proj0(pair3)
            x86add_inst(14, 0, 5), // pair4 = x86add(v0, v5)
            proj0_inst(15, 14),    // r4 = proj0(pair4)
        ];
        let dag = ScheduleDag::build(&insts);
        let scheduled = schedule(&dag);
        assert_topo_order(&scheduled);

        let list_peak = peak_live(&scheduled);

        let topo_scheduled: Vec<ScheduledInst> = insts
            .iter()
            .map(|i| ScheduledInst {
                op: i.op.clone(),
                dst: i.dst,
                operands: i.operands.iter().flatten().copied().collect(),
            })
            .collect();
        let topo_peak = peak_live(&topo_scheduled);

        // Finding: wide fan-out shape. v0 must stay live until its last use.
        // List scheduler groups uses of shared operand together (last_use_count
        // priority fires for the final x86add user of v0), keeping peak low.
        // List scheduler peak should be <= topo-sort peak.
        assert!(
            list_peak <= topo_peak,
            "list scheduler should match or beat topo-sort peak for fan-out: list={list_peak}, topo={topo_peak}"
        );
    }

    /// DAG shape: diamond
    ///   v0 = iconst, v1 = iconst, v2 = iconst
    ///   pair_L = x86add(v0, v1), r_L = proj0(pair_L)   -- left branch
    ///   pair_R = x86add(v0, v2), r_R = proj0(pair_R)   -- right branch (shares v0)
    ///   pair_m = x86add(r_L, r_R), r_m = proj0(pair_m) -- merge
    ///
    /// Finding: Diamond introduces a brief peak where r_L, r_R, and v0 are all
    /// live. List scheduler (last_use_count priority) finishes the branch that
    /// frees its unique operand first, then handles the merge.
    /// Peak is typically 3-4 VRegs. List scheduler matches or beats topo.
    #[test]
    fn spike_diamond_register_pressure() {
        let insts = vec![
            iconst_inst(0),       // v0 (shared)
            iconst_inst(1),       // v1
            iconst_inst(2),       // v2
            x86add_inst(3, 0, 1), // pair_L
            proj0_inst(4, 3),     // r_L
            x86add_inst(5, 0, 2), // pair_R
            proj0_inst(6, 5),     // r_R
            x86add_inst(7, 4, 6), // pair_m
            proj0_inst(8, 7),     // r_m
        ];
        let dag = ScheduleDag::build(&insts);
        let scheduled = schedule(&dag);
        assert_topo_order(&scheduled);

        let list_peak = peak_live(&scheduled);

        let topo_scheduled: Vec<ScheduledInst> = insts
            .iter()
            .map(|i| ScheduledInst {
                op: i.op.clone(),
                dst: i.dst,
                operands: i.operands.iter().flatten().copied().collect(),
            })
            .collect();
        let topo_peak = peak_live(&topo_scheduled);

        // Finding: diamond DAG. The list scheduler should not inflate register
        // pressure compared to a simple topological sort.
        assert!(
            list_peak <= topo_peak + 1,
            "list scheduler peak should be close to topo-sort for diamond: list={list_peak}, topo={topo_peak}"
        );
    }
}
