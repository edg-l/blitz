use std::collections::BTreeMap;

use smallvec::SmallVec;

use crate::egraph::eclass::EClass;
use crate::egraph::enode::ENode;
use crate::egraph::known_bits::KnownBits;
use crate::egraph::unionfind::UnionFind;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

/// Snapshot of an e-node for safe iteration during mutation.
#[derive(Clone)]
pub struct NodeSnap {
    pub class_id: ClassId,
    pub op: Op,
    pub children: SmallVec<[ClassId; 2]>,
}

/// Snapshot all canonical e-nodes for iteration.
pub fn snapshot_all(egraph: &EGraph) -> Vec<NodeSnap> {
    let mut snaps = Vec::new();
    for i in 0..egraph.classes.len() as u32 {
        let id = ClassId(i);
        if egraph.unionfind.find_immutable(id) != id {
            continue;
        }
        let class = egraph.class(id);
        for node in &class.nodes {
            snaps.push(NodeSnap {
                class_id: id,
                op: node.op.clone(),
                children: node.children.clone(),
            });
        }
    }
    snaps
}

#[derive(Clone)]
pub struct EGraph {
    pub(crate) unionfind: UnionFind,
    /// Arena: ClassId(i) indexes directly into classes[i].
    pub(crate) classes: Vec<EClass>,
    /// Hashcons: canonicalized ENode -> canonical ClassId.
    pub(crate) memo: BTreeMap<ENode, ClassId>,
    pub(crate) worklist: Vec<ClassId>,
    pub(crate) node_count: usize,
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    pub fn new() -> Self {
        Self {
            unionfind: UnionFind::new(),
            classes: Vec::new(),
            memo: BTreeMap::new(),
            worklist: Vec::new(),
            node_count: 0,
        }
    }

    /// Add an e-node to the e-graph, returning its e-class id.
    ///
    /// Canonicalizes children, checks the memo, and either returns the existing
    /// ClassId or creates a new EClass.
    pub fn add(&mut self, mut enode: ENode) -> ClassId {
        // Step 1: canonicalize children (skip NONE sentinel)
        for child in enode.children.iter_mut() {
            if *child != ClassId::NONE {
                *child = self.unionfind.find(*child);
            }
        }

        // Step 2: check memo
        if let Some(&existing) = self.memo.get(&enode) {
            return self.unionfind.find(existing);
        }

        // Step 3: compute result type from children's types.
        // ClassId::NONE is a sentinel for absent operands (e.g. Addr with no index);
        // we supply I64 as a placeholder type so result_type is not confused.
        let child_types: Vec<_> = enode
            .children
            .iter()
            .map(|c| {
                if *c == ClassId::NONE {
                    crate::ir::types::Type::I64
                } else {
                    self.classes[c.0 as usize].ty.clone()
                }
            })
            .collect();
        let ty = enode.op.result_type(&child_types);

        // Step 4: allocate new e-class
        let new_id = self.unionfind.make_set();
        assert_eq!(
            new_id.0 as usize,
            self.classes.len(),
            "ClassId must match classes index"
        );
        let constant_value = if let Op::Iconst(v, ref iconst_ty) = enode.op {
            Some((v, iconst_ty.clone()))
        } else {
            None
        };
        let known_bits = if let Op::Iconst(v, ref iconst_ty) = enode.op {
            KnownBits::from_constant(v, iconst_ty)
        } else {
            KnownBits::unknown()
        };
        let eclass = EClass {
            id: new_id,
            nodes: vec![enode.clone()],
            ty,
            constant_value,
            known_bits,
        };
        self.classes.push(eclass);
        self.memo.insert(enode, new_id);
        self.node_count += 1;

        new_id
    }

    /// Merge two e-classes. Panics if their types differ.
    ///
    /// Returns the canonical representative after merging.
    pub fn merge(&mut self, a: ClassId, b: ClassId) -> ClassId {
        let a = self.unionfind.find(a);
        let b = self.unionfind.find(b);

        if a == b {
            return a;
        }

        assert_eq!(
            self.classes[a.0 as usize].ty, self.classes[b.0 as usize].ty,
            "type mismatch when merging e-classes {:?} and {:?}",
            a, b
        );

        // Union in union-find; canonical is the winner
        let canonical = self.unionfind.union(a, b);
        let non_canonical = if canonical == a { b } else { a };

        // Read constant values and known-bits before mem::take moves nodes out of non-canonical
        let cv_a = self.classes[a.0 as usize].constant_value.clone();
        let cv_b = self.classes[b.0 as usize].constant_value.clone();
        let joined_cv = match (cv_a, cv_b) {
            (Some((va, ta)), Some((vb, _))) => {
                if va == vb {
                    Some((va, ta))
                } else {
                    // Different constants merged -- analysis result is unknown.
                    None
                }
            }
            (Some(cv), None) | (None, Some(cv)) => Some(cv),
            (None, None) => None,
        };
        let kb_a = self.classes[a.0 as usize].known_bits;
        let kb_b = self.classes[b.0 as usize].known_bits;
        let joined_kb = KnownBits::merge(&kb_a, &kb_b);

        // Move nodes from non-canonical into canonical
        // We need to split the borrow: collect the nodes first
        let moved_nodes: Vec<ENode> =
            std::mem::take(&mut self.classes[non_canonical.0 as usize].nodes);
        self.classes[canonical.0 as usize].nodes.extend(moved_nodes);
        self.classes[canonical.0 as usize].constant_value = joined_cv;
        self.classes[canonical.0 as usize].known_bits = joined_kb;

        self.worklist.push(canonical);
        canonical
    }

    /// Rebuild the e-graph to a consistent state (process the worklist).
    ///
    /// Re-canonicalizes all e-nodes across the entire memo until no non-canonical
    /// children remain and no further congruence merges are implied.
    pub fn rebuild(&mut self) {
        while !self.worklist.is_empty() {
            self.worklist.clear();

            // Drain the entire memo, re-canonicalize every node, re-insert.
            // Any collision means congruence: merge the two classes and add to worklist.
            let old_memo: BTreeMap<ENode, ClassId> = std::mem::take(&mut self.memo);

            // Also clear all class node lists so we can rebuild them clean
            for class in self.classes.iter_mut() {
                class.nodes.clear();
            }

            for (mut node, owner) in old_memo {
                // Re-canonicalize children (skip NONE sentinel)
                for child in node.children.iter_mut() {
                    if *child != ClassId::NONE {
                        *child = self.unionfind.find(*child);
                    }
                }
                let owner_canon = self.unionfind.find(owner);

                match self.memo.get(&node).copied() {
                    Some(existing) => {
                        let existing_canon = self.unionfind.find(existing);
                        if existing_canon != owner_canon {
                            // Congruence closure: merge and schedule for next round
                            self.classes[owner_canon.0 as usize].nodes.push(node);
                            self.merge(owner_canon, existing_canon);
                        } else {
                            self.classes[owner_canon.0 as usize].nodes.push(node);
                        }
                    }
                    None => {
                        self.memo.insert(node.clone(), owner_canon);
                        self.classes[owner_canon.0 as usize].nodes.push(node);
                    }
                }
            }
        }
    }

    /// Find the canonical representative of an e-class.
    pub fn find(&mut self, id: ClassId) -> ClassId {
        self.unionfind.find(id)
    }

    /// Get the constant value for an e-class, if known.
    /// Uses the e-class analysis rather than scanning nodes.
    pub fn get_constant(&self, class_id: ClassId) -> Option<(i64, Type)> {
        if class_id == ClassId::NONE {
            return None;
        }
        let canon = self.unionfind.find_immutable(class_id);
        self.classes[canon.0 as usize].constant_value.clone()
    }

    /// Look up the known-bits analysis for an e-class.
    pub fn get_known_bits(&self, class_id: ClassId) -> KnownBits {
        if class_id == ClassId::NONE {
            return KnownBits::unknown();
        }
        let canon = self.unionfind.find_immutable(class_id);
        self.classes[canon.0 as usize].known_bits
    }

    /// Look up an e-class by its canonical id.
    pub fn class(&self, id: ClassId) -> &EClass {
        &self.classes[id.0 as usize]
    }

    /// Number of live e-classes (those where find(id) == id).
    pub fn class_count(&self) -> usize {
        (0..self.classes.len() as u32)
            .filter(|&i| self.unionfind.find_immutable(ClassId(i)) == ClassId(i))
            .count()
    }

    /// Total number of e-nodes inserted (not decremented on merge).
    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn make_iconst(g: &mut EGraph, val: i64, ty: Type) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(val, ty),
            children: smallvec![],
        })
    }

    fn make_add(g: &mut EGraph, a: ClassId, b: ClassId) -> ClassId {
        g.add(ENode {
            op: Op::Add,
            children: smallvec![a, b],
        })
    }

    // 3.9 tests

    #[test]
    fn add_identical_nodes_returns_same_class() {
        let mut g = EGraph::new();
        let c1 = make_iconst(&mut g, 1, Type::I64);
        let c2 = make_iconst(&mut g, 2, Type::I64);
        let add1 = make_add(&mut g, c1, c2);
        let add2 = make_add(&mut g, c1, c2);
        assert_eq!(add1, add2);
    }

    #[test]
    fn add_different_nodes_returns_different_classes() {
        let mut g = EGraph::new();
        let c1 = make_iconst(&mut g, 1, Type::I64);
        let c2 = make_iconst(&mut g, 2, Type::I64);
        let c3 = make_iconst(&mut g, 3, Type::I64);
        let add1 = make_add(&mut g, c1, c2);
        let add2 = make_add(&mut g, c1, c3);
        assert_ne!(add1, add2);
    }

    #[test]
    fn merge_makes_find_equal() {
        let mut g = EGraph::new();
        let c1 = make_iconst(&mut g, 1, Type::I64);
        let c2 = make_iconst(&mut g, 2, Type::I64);
        assert_ne!(g.find(c1), g.find(c2));
        g.merge(c1, c2);
        assert_eq!(g.find(c1), g.find(c2));
    }

    #[test]
    fn rebuild_detects_congruence() {
        // add(a, b) in c1, add(a, b') in c2; merge b and b'; rebuild merges c1 and c2
        let mut g = EGraph::new();
        let a = make_iconst(&mut g, 10, Type::I64);
        let b = make_iconst(&mut g, 20, Type::I64);
        let b_prime = make_iconst(&mut g, 30, Type::I64);

        let c1 = make_add(&mut g, a, b);
        let c2 = make_add(&mut g, a, b_prime);
        assert_ne!(g.find(c1), g.find(c2));

        g.merge(b, b_prime);
        g.rebuild();

        assert_eq!(g.find(c1), g.find(c2));
    }

    // 3.10 stress test

    #[test]
    fn stress_10k_iconst_random_merges() {
        let mut g = EGraph::new();
        let mut ids: Vec<ClassId> = Vec::with_capacity(10_000);

        // Insert 10,000 Iconst nodes with different values
        for i in 0..10_000i64 {
            let id = make_iconst(&mut g, i, Type::I64);
            ids.push(id);
        }
        assert_eq!(g.node_count(), 10_000);

        // Perform deterministic "random" merges using a simple LCG
        let mut state: u64 = 0xdeadbeef_cafebabe;
        for _ in 0..500 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let i = (state >> 33) as usize % ids.len();
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % ids.len();
            if i != j {
                g.merge(ids[i], ids[j]);
            }
        }

        // rebuild must terminate
        g.rebuild();

        // Collect memo snapshot first (needs immutable borrow)
        let memo_snapshot: Vec<ENode> = g.memo.keys().cloned().collect();

        // Verify memo consistency: every entry's children are canonical
        let mut bad = 0;
        for node in &memo_snapshot {
            for &child in &node.children {
                if g.unionfind.find(child) != child {
                    bad += 1;
                }
            }
        }

        assert_eq!(g.node_count(), 10_000);
        assert_eq!(
            bad, 0,
            "memo has {} entries with non-canonical children",
            bad
        );
    }

    // Constant analysis tests

    #[test]
    fn constant_analysis_iconst_sets_value() {
        let mut g = EGraph::new();
        let c = make_iconst(&mut g, 42, Type::I64);
        assert_eq!(g.get_constant(c), Some((42, Type::I64)));
    }

    #[test]
    fn constant_analysis_non_const_is_none() {
        let mut g = EGraph::new();
        let a = make_iconst(&mut g, 1, Type::I64);
        let b = make_iconst(&mut g, 2, Type::I64);
        let add = make_add(&mut g, a, b);
        assert_eq!(g.get_constant(add), None);
    }

    #[test]
    fn constant_analysis_merge_same_preserves() {
        let mut g = EGraph::new();
        let a = make_iconst(&mut g, 42, Type::I64);
        let b = make_iconst(&mut g, 42, Type::I64);
        // a and b are already the same class via hash-consing
        assert_eq!(g.find(a), g.find(b));
        assert_eq!(g.get_constant(a), Some((42, Type::I64)));
    }

    #[test]
    fn constant_analysis_merge_const_with_non_const() {
        let mut g = EGraph::new();
        let c = make_iconst(&mut g, 42, Type::I64);
        let a = make_iconst(&mut g, 1, Type::I64);
        let b = make_iconst(&mut g, 2, Type::I64);
        let add = make_add(&mut g, a, b);
        // add has no constant; merge with c
        g.merge(add, c);
        let canon = g.find(add);
        assert_eq!(g.get_constant(canon), Some((42, Type::I64)));
    }

    #[test]
    fn constant_analysis_merge_different_consts_is_none() {
        let mut g = EGraph::new();
        let a = make_iconst(&mut g, 10, Type::I64);
        let b = make_iconst(&mut g, 20, Type::I64);
        g.merge(a, b);
        let canon = g.find(a);
        assert_eq!(g.get_constant(canon), None);
    }

    #[test]
    fn constant_analysis_survives_rebuild() {
        let mut g = EGraph::new();
        let c = make_iconst(&mut g, 99, Type::I64);
        g.rebuild();
        assert_eq!(g.get_constant(c), Some((99, Type::I64)));
    }
}
