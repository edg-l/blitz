use super::interference::InterferenceGraph;

/// Aggressive coalescing on the SSA interference graph.
///
/// For each copy pair `(src, dst)`, if src and dst do not interfere,
/// they can be merged (assigned the same physical register).
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
) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = Vec::new();
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
            continue;
        }

        // Check if the two representative groups interfere. `adj` is kept in
        // sync with merges, so this considers every member of either group.
        if adj[src_root].contains(&dst_root) || adj[dst_root].contains(&src_root) {
            continue;
        }

        // Different register classes must never coalesce (GPR <-> XMM merge
        // is always invalid regardless of adjacency).
        if graph.reg_class[src_root] != graph.reg_class[dst_root] {
            continue;
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
        let result = coalesce(&graph, &pairs);
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
        let result = coalesce(&graph, &pairs);
        assert!(result.is_empty(), "interfering pair must not be coalesced");
    }

    // Multiple non-interfering pairs all coalesced.
    #[test]
    fn multiple_non_interfering_coalesced() {
        let graph = make_graph(4, &[]);
        let pairs = [(0, 1), (2, 3)];
        let result = coalesce(&graph, &pairs);
        assert_eq!(result.len(), 2);
    }
}
