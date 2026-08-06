use std::collections::BTreeMap;

use super::interference::InterferenceGraph;
use crate::egraph::extract::VReg;
use crate::x86::reg::RegClass;

/// Follow a coalescing alias chain to the VReg that survived the merge.
///
/// The map is transitive, and stopping after one step leaves a VReg with no
/// register assignment -- which reads as "no answer" and drops whatever copy was
/// being emitted. The step count is bounded by the map's size so a cycle
/// terminates instead of hanging the compiler.
pub fn chase_alias(mut vreg: VReg, coalesce_aliases: &BTreeMap<VReg, VReg>) -> VReg {
    for _ in 0..coalesce_aliases.len() + 1 {
        match coalesce_aliases.get(&vreg) {
            Some(&aliased) if aliased != vreg => vreg = aliased,
            _ => break,
        }
    }
    vreg
}

/// Conservative (Briggs) coalescing on the SSA interference graph.
///
/// For each copy pair `(src, dst)`, if src and dst do not interfere and the
/// merged node would still be colorable, they are merged (assigned the same
/// physical register).
///
/// Non-interference alone is not enough. Merging replaces two nodes with one
/// whose neighbourhood is the union of theirs, so a merge that is individually
/// legal can still raise the chromatic number above the register budget. Six
/// parameters of one nine-parameter block, each merged onto a different constant
/// living across the whole function, built a 15-clique against a 14-register
/// budget out of merges that all passed the interference test.
///
/// The Briggs test admits a merge only when the merged node has fewer than `k`
/// neighbours of significant degree (degree >= `k`), counting per register class.
/// Nodes below that degree can always be colored after their neighbours, so they
/// cannot be what makes the graph uncolorable.
///
/// Must be run on the original SSA graph BEFORE spill code insertion.
/// After spill insertion the graph may not be chordal, so coalescing
/// must not be re-run.
///
/// Returns a list of `(merged_into, merged_from)` pairs: the `merged_from`
/// VReg should be treated as an alias for `merged_into` everywhere.
pub fn coalesce(
    graph: &InterferenceGraph,
    copy_pairs: &[(usize, usize)], // (src, dst) VReg indices
    gpr_colors: u32,
    xmm_colors: u32,
) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = Vec::new();
    // Why each candidate copy was refused, for the summary trace below.
    let mut why: std::collections::BTreeMap<&str, usize> = Default::default();
    // Union-find to track already-merged groups.
    let mut parent: Vec<usize> = (0..graph.num_vregs).collect();

    // Per-root adjacency: when two nodes are merged, their adjacency sets are
    // unioned into the surviving root. This is required for correctness —
    // otherwise, a post-merge coalesce check only inspects the root's
    // original adj[], missing interferences that belonged to the merged
    // member. Concretely: if v0 coalesces with v6, and v6 interferes with
    // v9, then v9 must not coalesce with v0. Without union, `adj[v0]` never
    // learned about v6's interference with v9 and the second coalesce
    // succeeds incorrectly.
    let mut adj: Vec<std::collections::BTreeSet<usize>> = graph.adj.clone();

    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        x
    };

    for &(src, dst) in copy_pairs {
        if src >= graph.num_vregs || dst >= graph.num_vregs {
            continue;
        }

        let src_root = find(&mut parent, src);
        let dst_root = find(&mut parent, dst);

        if src_root == dst_root {
            // Already in the same group.
            *why.entry("already merged").or_default() += 1;
            continue;
        }

        // Check if the two representative groups interfere. `adj` is kept in
        // sync with merges, so this considers every member of either group.
        if adj[src_root].contains(&dst_root) || adj[dst_root].contains(&src_root) {
            *why.entry("groups interfere").or_default() += 1;
            continue;
        }

        // Different register classes must never coalesce (GPR <-> XMM merge
        // is always invalid regardless of adjacency).
        if graph.reg_class[src_root] != graph.reg_class[dst_root] {
            *why.entry("different reg class").or_default() += 1;
            continue;
        }

        // Briggs: the merged node must have fewer than k neighbours of
        // significant degree. Degrees are read from `adj`, which merges keep
        // current, so this measures the graph as it stands.
        let k = match graph.reg_class[src_root] {
            RegClass::GPR => gpr_colors,
            RegClass::XMM => xmm_colors,
            // One EFLAGS. Two flags values never live at once, so nothing is
            // ever a candidate here, and if one were, k = 1 refuses it.
            RegClass::Flags => 1,
        } as usize;
        let class = graph.reg_class[src_root];
        let mut significant: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for &n in adj[src_root].iter().chain(adj[dst_root].iter()) {
            let n_root = find(&mut parent, n);
            if n_root == src_root || n_root == dst_root {
                continue;
            }
            if graph.reg_class[n_root] == class && adj[n_root].len() >= k {
                significant.insert(n_root);
            }
        }
        // Briggs is conservative and refuses most merges here: 39 of 64
        // candidate copies on `queens`, 44 of 112 on `hash_table`. George's
        // test admits a different set and is conservative in the same sense --
        // it passes only when the merge constrains no neighbour that was not
        // already constrained -- so taking either is still safe.
        //
        // For every neighbour `t` of the node being merged away: `t` is
        // harmless if it cannot compete for the colour (different class), if it
        // already interferes with the survivor, or if it has room to spare
        // (degree below k). If that holds of all of them, the survivor's
        // neighbourhood gains nothing that can stop it being coloured.
        //
        // Checked in both orientations: either one passing justifies the same
        // merged node.
        let george = |a: usize, b: usize, parent: &mut Vec<usize>| {
            adj[a].iter().all(|&t| {
                let t_root = find(parent, t);
                t_root == a
                    || t_root == b
                    || graph.reg_class[t_root] != class
                    || adj[b].contains(&t_root)
                    || adj[t_root].len() < k
            })
        };
        if significant.len() >= k
            && !george(dst_root, src_root, &mut parent)
            && !george(src_root, dst_root, &mut parent)
        {
            *why.entry("briggs and george both decline").or_default() += 1;
            continue;
        }

        if crate::trace::is_enabled("coalesce") {
            // The merge is between the two groups' representatives, not between
            // the copy's own endpoints: `src`/`dst` name the copy, `src_root`/
            // `dst_root` name everything that goes with them. A merge whose
            // endpoints carry no edge in the pre-merge graph while its groups
            // hold values that do overlap is how an illegal merge gets in, and
            // the degrees say whether the Briggs test was what admitted it.
            tracing::debug!(
                target: "blitz::coalesce",
                "merge v{src_root} <- v{dst_root} for copy (v{src}, v{dst}); \
                 pre_merge_edge={} degrees {}/{} against k={k}",
                graph.adj[src_root].contains(&dst_root),
                adj[src_root].len(),
                adj[dst_root].len(),
            );
        }

        // Coalesce: merge dst_root into src_root. Transfer adjacency so
        // subsequent checks against src_root see dst_root's neighbors too.
        // For every neighbor n of dst_root, update adj[n] to reference
        // src_root (via their current roots) and add to adj[src_root].
        let dst_neighbors: Vec<usize> = adj[dst_root].iter().copied().collect();
        for n in dst_neighbors {
            let n_root = find(&mut parent, n);
            if n_root == src_root {
                // The merged pair was both neighbors of src_root already —
                // cannot happen here since we checked non-interference above,
                // but skip defensively.
                continue;
            }
            adj[src_root].insert(n_root);
            adj[n_root].insert(src_root);
        }
        adj[dst_root].clear();
        parent[dst_root] = src_root;
        merged.push((src_root, dst_root));
    }

    // What was declined and why. The trace named only the merges it made, so a
    // copy surviving into the emitted code had no explanation anywhere -- and
    // copies are a third of what this backend emits.
    if crate::trace::is_enabled("coalesce") {
        tracing::debug!(
            target: "blitz::coalesce",
            "{} copy pairs, {} merged; declined: {:?}",
            copy_pairs.len(),
            merged.len(),
            why,
        );
    }
    merged
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

    // Non-interfering copy pair is coalesced.
    #[test]
    fn non_interfering_pair_coalesced() {
        let graph = make_graph(3, &[(0, 2)]); // v0--v2 interfere; v1 is isolated
        // Copy pair: v1 -> v0 (src=1, dst=0). They don't interfere.
        let pairs = [(1, 0)];
        let result = coalesce(&graph, &pairs, 14, 16);
        assert_eq!(result.len(), 1, "one coalescing merge expected");
        // Either (0,1) or (1,0) depending on merge direction.
        let (into, from) = result[0];
        assert!(
            (into == 0 && from == 1) || (into == 1 && from == 0),
            "unexpected merge: ({into}, {from})"
        );
    }

    // Interfering copy pair is NOT coalesced.
    #[test]
    fn interfering_pair_not_coalesced() {
        // v5 and v3 interfere.
        let graph = make_graph(6, &[(3, 5)]);
        let pairs = [(3, 5)]; // copy pair between interfering VRegs
        let result = coalesce(&graph, &pairs, 14, 16);
        assert!(result.is_empty(), "interfering pair must not be coalesced");
    }

    // Multiple non-interfering pairs all coalesced.
    #[test]
    fn multiple_non_interfering_coalesced() {
        let graph = make_graph(4, &[]);
        let pairs = [(0, 1), (2, 3)];
        let result = coalesce(&graph, &pairs, 14, 16);
        assert_eq!(result.len(), 2);
    }

    // A merge whose result would have k neighbours of significant degree is
    // declined even though the pair does not interfere, and the same merge is
    // taken when the budget is large enough to color around it.
    #[test]
    fn briggs_declines_merge_that_raises_degree() {
        // v0 and v1 do not interfere. v0's neighbour v2 and v1's neighbour v3
        // both have degree 3, so with k=2 they are both significant and the
        // merged node would have 2 >= k of them.
        let graph = make_graph(6, &[(0, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 5)]);
        let pairs = [(0, 1)];

        assert!(
            coalesce(&graph, &pairs, 2, 2).is_empty(),
            "merge must be declined when the merged node has k significant neighbours"
        );
        assert_eq!(
            coalesce(&graph, &pairs, 14, 16).len(),
            1,
            "the same merge is safe against a budget no degree here reaches"
        );
    }
}
