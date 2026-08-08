use std::collections::BTreeMap;

use super::interference::InterferenceGraph;
use crate::egraph::extract::VReg;
use crate::regalloc::interference::VRegSet;
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

/// Optimistic coalescing on the SSA interference graph.
///
/// For each copy pair `(src, dst)`, if the two groups do not interfere they are
/// merged and the copy disappears. **Non-interference is the only test.**
///
/// Merging replaces two nodes with one whose neighbourhood is the union of
/// theirs, so a merge that is individually legal can still raise the chromatic
/// number above the register budget -- six parameters of one nine-parameter
/// block, each merged onto a different constant living across the whole
/// function, built a 15-clique against a 14-register budget out of merges that
/// all passed the interference test. The conservative answer to that is Briggs
/// and George, which admit a merge only when the merged node provably cannot be
/// what makes the graph uncolourable.
///
/// **Both were in, and measurement retired them.** Over the `live` kernels they
/// refused 986 of 2476 candidate copies while only 5 pairs genuinely interfered:
/// the refusals were almost entirely the conservative tests declining to
/// predict, not merges that were unsafe. Merging anyway and undoing what does
/// not colour is Park & Moon's optimistic coalescing, and it is worth `-28%`
/// copies and `-7%` instructions across the corpora.
///
/// So the colourability question moved to where the answer is known.
/// `deny` is how the caller states it: a pair in that set is not merged, and
/// `allocate_global` grows it from the VRegs a failed colouring names. This
/// function no longer guesses, and does not need to be conservative, because a
/// wrong guess is now retried rather than emitted.
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
    deny: &std::collections::BTreeSet<(usize, usize)>,
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
    let mut adj: Vec<VRegSet> = graph.adj.clone();

    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        x
    };

    // One pass, in the caller's order.
    //
    // The fixpoint loop this replaced existed for one reason: Briggs and George
    // are stated against significant degree, and merging two nodes drops the
    // degree of every neighbour they shared, so a copy refused against the graph
    // as it stands can be safe one merge later. With no conservative test left
    // there is nothing a second round could overturn -- interference only ever
    // grows as groups merge, so a pair refused for interference stays refused.
    //
    // Note this is *not* George & Appel's iterated coalescing, which measurement
    // separately retired: their leverage is `simplify`, and simulated over the
    // `bench` kernels it removes 62 of 165 nodes on `queens` and 80 of 297 on
    // `hash_table` while letting through **zero** additional copies -- the
    // survivors have their endpoints in the dense core, which simplification
    // does not touch.
    let candidates: Vec<(usize, usize)> = copy_pairs
        .iter()
        .copied()
        .filter(|&(src, dst)| src < graph.num_vregs && dst < graph.num_vregs)
        .collect();
    {
        for (src, dst) in candidates.iter().copied() {
            let src_root = find(&mut parent, src);
            let dst_root = find(&mut parent, dst);

            if src_root == dst_root {
                // Already in the same group; nothing will ever change that.
                *why.entry("already merged").or_default() += 1;
                continue;
            }

            // A merge a previous colouring attempt blamed for its overshoot.
            if deny.contains(&(src, dst)) {
                *why.entry("denied by a failed colouring").or_default() += 1;
                continue;
            }

            // Whether the two representative groups interfere. `adj` is kept in
            // sync with merges, so this considers every member of either group,
            // and interference only ever grows as groups merge.
            if adj[src_root].contains(dst_root) || adj[dst_root].contains(src_root) {
                *why.entry("groups interfere").or_default() += 1;
                continue;
            }

            // Different register classes must never coalesce (GPR <-> XMM merge
            // is always invalid regardless of adjacency).
            if graph.reg_class[src_root] != graph.reg_class[dst_root] {
                *why.entry("different reg class").or_default() += 1;
                continue;
            }

            // One EFLAGS, and no store reaches it, so a flags value can never
            // share a register with anything: two of them are never live at
            // once, and merging one onto a value that is would put a
            // comparison's result where the other value has to be.
            if graph.reg_class[src_root] == RegClass::Flags {
                *why.entry("flags").or_default() += 1;
                continue;
            }
            if crate::trace::is_enabled("coalesce") {
                // The merge is between the two groups' representatives, not
                // between the copy's own endpoints: `src`/`dst` name the copy,
                // `src_root`/`dst_root` name everything that goes with them. A
                // merge whose endpoints carry no edge in the pre-merge graph
                // while its groups hold values that do overlap is how an
                // illegal merge gets in.
                tracing::debug!(
                    target: "blitz::coalesce",
                    "merge v{src_root} <- v{dst_root} for copy (v{src}, v{dst}); \
                     pre_merge_edge={} degrees {}/{}",
                    graph.adj[src_root].contains(dst_root),
                    adj[src_root].len(),
                    adj[dst_root].len(),
                );
            }

            // Merge dst_root into src_root. Transfer adjacency so subsequent
            // checks against src_root see dst_root's neighbours too.
            let dst_neighbors: Vec<usize> = adj[dst_root].iter().collect();
            for n in dst_neighbors {
                let n_root = find(&mut parent, n);
                if n_root == src_root {
                    // Cannot happen -- non-interference was checked above --
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
    use crate::regalloc::interference::{InterferenceGraph, VRegSet};
    use crate::x86::reg::RegClass;
    use std::collections::BTreeSet;

    fn make_graph(n: usize, edges: &[(usize, usize)]) -> InterferenceGraph {
        let mut g = InterferenceGraph {
            num_vregs: n,
            adj: vec![VRegSet::new(); n],
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
        let result = coalesce(&graph, &pairs, &BTreeSet::new());
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
        let result = coalesce(&graph, &pairs, &BTreeSet::new());
        assert!(result.is_empty(), "interfering pair must not be coalesced");
    }

    // Multiple non-interfering pairs all coalesced.
    #[test]
    fn multiple_non_interfering_coalesced() {
        let graph = make_graph(4, &[]);
        let pairs = [(0, 1), (2, 3)];
        let result = coalesce(&graph, &pairs, &BTreeSet::new());
        assert_eq!(result.len(), 2);
    }

    // A merge that raises the merged node's degree is taken anyway: whether the
    // graph can still be coloured is the colourer's answer, not a prediction
    // made here, and `allocate_global` denies the merge and retries if it was
    // wrong.
    #[test]
    fn a_merge_that_raises_degree_is_taken_anyway() {
        // v0 and v1 do not interfere. v0's neighbour v2 and v1's neighbour v3
        // both have degree 3, which is what the Briggs test used to refuse on.
        let graph = make_graph(6, &[(0, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 5)]);
        let pairs = [(0, 1)];
        assert_eq!(coalesce(&graph, &pairs, &BTreeSet::new()).len(), 1);
    }

    // The denylist is how a failed colouring takes a merge back.
    #[test]
    fn a_denied_pair_is_not_merged() {
        let graph = make_graph(3, &[(0, 2)]);
        let pairs = [(1, 0)];
        let deny: BTreeSet<(usize, usize)> = [(1, 0)].into_iter().collect();
        assert!(
            coalesce(&graph, &pairs, &deny).is_empty(),
            "a denied pair must survive as a copy"
        );
    }

    // A denied pair does not block an unrelated one.
    #[test]
    fn denying_one_pair_leaves_the_others() {
        let graph = make_graph(4, &[]);
        let pairs = [(0, 1), (2, 3)];
        let deny: BTreeSet<(usize, usize)> = [(0, 1)].into_iter().collect();
        assert_eq!(coalesce(&graph, &pairs, &deny).len(), 1);
    }
}
