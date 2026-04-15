use std::collections::BTreeMap;

use crate::x86::abi::{CALLEE_SAVED, CALLER_SAVED_GPR};
use crate::x86::reg::{Reg, RegClass};

use super::interference::InterferenceGraph;

// ── MCS ordering (10.4) ───────────────────────────────────────────────────────

/// Maximum Cardinality Search: process vertices in order of maximum
/// already-processed neighbors. Returns a simplicial elimination ordering.
///
/// For chordal graphs (which SSA interference graphs are), greedy coloring
/// in reverse MCS order is optimal.
pub fn mcs_ordering(graph: &InterferenceGraph) -> Vec<usize> {
    let n = graph.num_vregs;
    if n == 0 {
        return vec![];
    }

    // weight[v] = number of already-processed neighbors of v.
    let mut weight = vec![0usize; n];
    let mut processed = vec![false; n];
    let mut ordering = Vec::with_capacity(n);

    for _ in 0..n {
        // Pick the unprocessed vertex with the highest weight.
        // On ties, pick the one with the smallest index (stable).
        let v = (0..n)
            .filter(|&i| !processed[i])
            .max_by_key(|&i| (weight[i], usize::MAX - i))
            .expect("at least one unprocessed vertex");

        processed[v] = true;
        ordering.push(v);

        // Update weights of unprocessed neighbors.
        for &neighbor in &graph.adj[v] {
            if !processed[neighbor] {
                weight[neighbor] += 1;
            }
        }
    }

    ordering
}

// ── Greedy coloring (10.5) ────────────────────────────────────────────────────

pub struct ColoringResult {
    /// VReg index -> color (None if the VReg has no neighbors and no pre-color,
    /// but in practice all VRegs get a color).
    pub colors: Vec<Option<u32>>,
    /// The number of distinct colors used.
    pub chromatic_number: u32,
}

/// Greedy graph coloring in reverse MCS order.
///
/// Colors VRegs with the smallest available color not used by any
/// already-colored neighbor. Respects `pre_coloring` constraints.
pub fn greedy_color(
    graph: &InterferenceGraph,
    ordering: &[usize],
    pre_coloring: &BTreeMap<usize, u32>,
) -> ColoringResult {
    let n = graph.num_vregs;
    let mut colors: Vec<Option<u32>> = vec![None; n];

    // Apply pre-colorings first.
    for (&vreg, &color) in pre_coloring {
        if vreg < n {
            colors[vreg] = Some(color);
        }
    }

    // Color in reverse MCS order (gives optimal coloring on chordal graphs).
    for &v in ordering.iter().rev() {
        if colors[v].is_some() {
            // Already pre-colored; skip.
            continue;
        }

        // Collect colors used by already-colored neighbors.
        let mut forbidden: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for &neighbor in &graph.adj[v] {
            if let Some(c) = colors[neighbor] {
                forbidden.insert(c);
            }
        }

        // Assign smallest non-forbidden color.
        let mut color = 0u32;
        while forbidden.contains(&color) {
            color += 1;
        }
        colors[v] = Some(color);
    }

    // Handle any VRegs not in the ordering (isolated nodes with no edges).
    // Give them color 0 if not already colored.
    for color in colors.iter_mut().take(n) {
        if color.is_none() {
            *color = Some(0);
        }
    }

    let chromatic_number = colors
        .iter()
        .filter_map(|&c| c)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    ColoringResult {
        colors,
        chromatic_number,
    }
}

/// Interval-based coloring for single-block SSA.
///
/// Processes instructions left-to-right, assigning each dst the smallest color
/// not used by any currently-alive VReg. Guaranteed optimal for interval graphs
/// (chromatic number = max clique = max simultaneous liveness + 1).
///
/// Falls back gracefully when the graph is not a perfect interval graph.
pub fn interval_color(
    insts: &[crate::schedule::scheduler::ScheduledInst],
    liveness: &super::liveness::LivenessInfo,
    pre_coloring: &BTreeMap<usize, u32>,
    num_vregs: usize,
) -> ColoringResult {
    let mut colors: Vec<Option<u32>> = vec![None; num_vregs];

    // Apply pre-colorings first.
    for (&vreg, &color) in pre_coloring {
        if vreg < num_vregs {
            colors[vreg] = Some(color);
        }
    }

    // Process instructions in order. At each instruction, the dst needs a color
    // that doesn't conflict with any VReg in live_at[i] (alive before this inst).
    for (i, inst) in insts.iter().enumerate() {
        let dst = inst.dst.0 as usize;
        if dst >= num_vregs || colors[dst].is_some() {
            continue;
        }
        // Forbidden: colors of VRegs alive before this instruction.
        let mut forbidden = std::collections::BTreeSet::new();
        if i < liveness.live_at.len() {
            for v in &liveness.live_at[i] {
                let idx = v.0 as usize;
                if idx < num_vregs
                    && let Some(c) = colors[idx]
                {
                    forbidden.insert(c);
                }
            }
        }
        let mut color = 0u32;
        while forbidden.contains(&color) {
            color += 1;
        }
        colors[dst] = Some(color);
    }

    // Any uncolored VRegs (not in instruction list, or in live_out only).
    for c in colors.iter_mut().take(num_vregs) {
        if c.is_none() {
            *c = Some(0);
        }
    }

    let chromatic_number = colors
        .iter()
        .filter_map(|&c| c)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    ColoringResult {
        colors,
        chromatic_number,
    }
}

// ── Physical register mapping (10.8) ─────────────────────────────────────────

/// Map color numbers to physical registers.
///
/// Excludes RSP. Pre-colored VRegs fix certain color->reg mappings.
/// For remaining colors: assigns caller-saved first, then callee-saved.
/// When `uses_frame_pointer` is false, RBP is included as an allocatable GPR.
pub fn map_colors_to_regs(
    coloring: &ColoringResult,
    reg_class: RegClass,
    pre_coloring: &BTreeMap<usize, Reg>,
    uses_frame_pointer: bool,
) -> BTreeMap<u32, Reg> {
    let mut color_to_reg: BTreeMap<u32, Reg> = BTreeMap::new();

    // First, establish color->reg from pre-colored VRegs.
    for (&vreg_idx, &reg) in pre_coloring {
        if vreg_idx < coloring.colors.len()
            && let Some(color) = coloring.colors[vreg_idx]
        {
            color_to_reg.insert(color, reg);
        }
    }

    // Build an ordered list of available physical registers for this class.
    // Prefer caller-saved first (avoids callee-save push/pop overhead).
    let available: Vec<Reg> = match reg_class {
        RegClass::GPR => allocatable_gpr_order(uses_frame_pointer),
        RegClass::XMM => allocatable_xmm_order(),
    };

    // Track which physical registers are already claimed.
    let claimed: std::collections::BTreeSet<Reg> = color_to_reg.values().copied().collect();
    let mut free_regs: Vec<Reg> = available
        .iter()
        .filter(|r| !claimed.contains(r))
        .copied()
        .collect();
    let mut free_iter = free_regs.drain(..);

    // Assign remaining colors.
    let max_color = coloring.chromatic_number;
    for color in 0..max_color {
        if color_to_reg.contains_key(&color) {
            continue;
        }
        if let Some(reg) = free_iter.next() {
            color_to_reg.insert(color, reg);
        }
        // If no register is available, the caller should have detected this
        // via chromatic_number > available_regs and triggered spilling.
    }

    color_to_reg
}

// ── Available register counts ─────────────────────────────────────────────────

/// Returns the ordered list of allocatable GPR registers.
///
/// Caller-saved registers are listed first (cheaper: no push/pop needed).
/// Callee-saved registers follow. When `uses_frame_pointer` is false, RBP is
/// appended last as a last-resort callee-saved general-purpose register.
///
/// This ordering must be kept consistent across `map_colors_to_regs`,
/// `build_pre_coloring_colors`, and `add_call_clobber_interferences` so that
/// color numbers correspond to the same physical register in all uses.
pub fn allocatable_gpr_order(uses_frame_pointer: bool) -> Vec<Reg> {
    // Caller-saved GPRs (no RSP).
    let mut regs: Vec<Reg> = CALLER_SAVED_GPR
        .iter()
        .filter(|&&r| r != Reg::RSP)
        .copied()
        .collect();
    // Callee-saved GPRs (RBP is excluded from the fixed list here; added below conditionally).
    regs.extend(CALLEE_SAVED.iter().filter(|&&r| r != Reg::RBP).copied());
    if !uses_frame_pointer {
        // When the frame pointer is omitted, RBP becomes an ordinary callee-saved GPR.
        // It is placed last so it is used only when all other registers are exhausted.
        regs.push(Reg::RBP);
    }
    regs
}

/// Number of usable GPR colors when the frame pointer is used (RBP reserved): 14.
/// Number of usable GPR colors when the frame pointer is omitted (RBP allocatable): 15.
const GPR_COLORS_WITH_FP: u32 = 14;
const GPR_COLORS_NO_FP: u32 = 15;

pub fn available_gpr_colors(uses_frame_pointer: bool) -> u32 {
    if uses_frame_pointer {
        GPR_COLORS_WITH_FP
    } else {
        GPR_COLORS_NO_FP
    }
}

/// Number of usable XMM colors (16 XMM registers).
pub const AVAILABLE_XMM_COLORS: u32 = 16;

/// Returns the ordered list of allocatable XMM registers (XMM0-XMM15).
///
/// All XMM registers are caller-saved in SysV ABI (no callee-saved XMMs).
/// This ordering must be kept consistent with `map_colors_to_regs` and
/// `add_xmm_call_clobber_interferences` so that color numbers correspond
/// to the correct physical registers.
pub fn allocatable_xmm_order() -> Vec<Reg> {
    vec![
        Reg::XMM0,
        Reg::XMM1,
        Reg::XMM2,
        Reg::XMM3,
        Reg::XMM4,
        Reg::XMM5,
        Reg::XMM6,
        Reg::XMM7,
        Reg::XMM8,
        Reg::XMM9,
        Reg::XMM10,
        Reg::XMM11,
        Reg::XMM12,
        Reg::XMM13,
        Reg::XMM14,
        Reg::XMM15,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc::interference::InterferenceGraph;
    use crate::x86::reg::RegClass;

    fn make_graph(n: usize, edges: &[(usize, usize)]) -> InterferenceGraph {
        let mut g = InterferenceGraph {
            num_vregs: n,
            adj: vec![std::collections::BTreeSet::new(); n],
            reg_class: vec![RegClass::GPR; n],
        };
        for &(a, b) in edges {
            g.add_edge(a, b);
        }
        g
    }

    // 10.6: No interference -> each gets color 0.
    #[test]
    fn no_interference_all_color_zero() {
        let graph = make_graph(4, &[]);
        let ordering = mcs_ordering(&graph);
        let result = greedy_color(&graph, &ordering, &BTreeMap::new());

        for c in &result.colors {
            assert_eq!(*c, Some(0), "all isolated nodes should get color 0");
        }
        assert_eq!(result.chromatic_number, 1);
    }

    // 10.6: Chain of interferences -> 2 colors.
    // v0 -- v1 -- v2: need 2 colors (alternating).
    #[test]
    fn chain_two_colors() {
        let graph = make_graph(3, &[(0, 1), (1, 2)]);
        let ordering = mcs_ordering(&graph);
        let result = greedy_color(&graph, &ordering, &BTreeMap::new());

        // Adjacent nodes must have different colors.
        assert_ne!(result.colors[0], result.colors[1]);
        assert_ne!(result.colors[1], result.colors[2]);
        // Chain is 2-colorable.
        assert!(result.chromatic_number <= 2, "chain needs at most 2 colors");
    }

    // 10.6: Clique of size 4 -> 4 colors.
    #[test]
    fn clique_four_colors() {
        let edges = &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let graph = make_graph(4, edges);
        let ordering = mcs_ordering(&graph);
        let result = greedy_color(&graph, &ordering, &BTreeMap::new());

        // All nodes in a clique must have distinct colors.
        let c: Vec<u32> = result.colors.iter().map(|c| c.unwrap()).collect();
        let mut unique = c.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 4, "K4 needs exactly 4 colors");
        assert_eq!(result.chromatic_number, 4);
    }

    // 10.7: Pre-coloring: first parameter gets RDI (color fixed).
    #[test]
    fn pre_coloring_respected() {
        // Two nodes with no interference; v0 pre-colored to color 5.
        let graph = make_graph(2, &[]);
        let ordering = mcs_ordering(&graph);
        let mut pre = BTreeMap::new();
        pre.insert(0usize, 5u32); // v0 -> color 5
        let result = greedy_color(&graph, &ordering, &pre);

        assert_eq!(result.colors[0], Some(5), "pre-coloring must be respected");
    }

    // Pre-coloring with interference: neighbors of pre-colored node avoid that color.
    #[test]
    fn pre_coloring_propagates() {
        // v0 -- v1; v0 pre-colored to color 0.
        let graph = make_graph(2, &[(0, 1)]);
        let ordering = mcs_ordering(&graph);
        let mut pre = BTreeMap::new();
        pre.insert(0usize, 0u32);
        let result = greedy_color(&graph, &ordering, &pre);

        assert_eq!(result.colors[0], Some(0));
        assert_ne!(
            result.colors[1],
            Some(0),
            "v1 must not use same color as v0"
        );
    }

    // map_colors_to_regs: basic smoke test.
    #[test]
    fn color_to_reg_basic() {
        let graph = make_graph(2, &[(0, 1)]);
        let ordering = mcs_ordering(&graph);
        let result = greedy_color(&graph, &ordering, &BTreeMap::new());
        let color_to_reg = map_colors_to_regs(&result, RegClass::GPR, &BTreeMap::new(), true);

        // Two colors, two distinct registers.
        let r0 = color_to_reg[&0];
        let r1 = color_to_reg[&1];
        assert_ne!(r0, r1, "different colors must map to different registers");
        assert!(r0.is_gpr() && r1.is_gpr());
        assert_ne!(r0, Reg::RSP);
        assert_ne!(r1, Reg::RSP);
    }

    // map_colors_to_regs: RBP is not allocated when uses_frame_pointer=true.
    #[test]
    fn color_to_reg_no_rbp_with_frame_pointer() {
        let graph = make_graph(2, &[(0, 1)]);
        let ordering = mcs_ordering(&graph);
        let result = greedy_color(&graph, &ordering, &BTreeMap::new());
        let color_to_reg = map_colors_to_regs(&result, RegClass::GPR, &BTreeMap::new(), true);
        for &reg in color_to_reg.values() {
            assert_ne!(
                reg,
                Reg::RBP,
                "RBP must not be allocated when frame pointer is used"
            );
        }
    }

    // map_colors_to_regs: RBP can be allocated when uses_frame_pointer=false.
    #[test]
    fn color_to_reg_rbp_available_without_frame_pointer() {
        // available_gpr_colors(false) = 15; if we need exactly 15 colors, color 14 -> RBP.
        assert_eq!(available_gpr_colors(false), 15);
        assert_eq!(available_gpr_colors(true), 14);
        let order = allocatable_gpr_order(false);
        assert_eq!(order.last(), Some(&Reg::RBP));
        let order_fp = allocatable_gpr_order(true);
        assert!(!order_fp.contains(&Reg::RBP));
    }
}
