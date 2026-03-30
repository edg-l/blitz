use crate::ir::op::ClassId;

/// Union-find with path compression and union by rank.
pub struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl Default for UnionFind {
    fn default() -> Self {
        Self::new()
    }
}

impl UnionFind {
    pub fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    /// Create a new singleton set; returns its ClassId.
    pub fn make_set(&mut self) -> ClassId {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        ClassId(id)
    }

    /// Find the canonical representative without path compression (immutable borrow).
    /// Slightly slower than `find`, but safe when only a shared reference is available.
    pub fn find_immutable(&self, id: ClassId) -> ClassId {
        if id == ClassId::NONE {
            return ClassId::NONE;
        }
        let mut x = id.0 as usize;
        while self.parent[x] as usize != x {
            x = self.parent[x] as usize;
        }
        ClassId(x as u32)
    }

    /// Find the canonical representative with path compression.
    pub fn find(&mut self, id: ClassId) -> ClassId {
        if id == ClassId::NONE {
            return ClassId::NONE;
        }
        let mut x = id.0 as usize;
        // Walk to root
        while self.parent[x] as usize != x {
            // Path compression: point directly to grandparent
            let gp = self.parent[self.parent[x] as usize];
            self.parent[x] = gp;
            x = gp as usize;
        }
        ClassId(x as u32)
    }

    /// Union two sets by rank. Returns the canonical representative.
    pub fn union(&mut self, a: ClassId, b: ClassId) -> ClassId {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let rank_a = self.rank[ra.0 as usize];
        let rank_b = self.rank[rb.0 as usize];
        // Attach smaller rank tree under larger rank tree
        if rank_a < rank_b {
            self.parent[ra.0 as usize] = rb.0;
            rb
        } else if rank_a > rank_b {
            self.parent[rb.0 as usize] = ra.0;
            ra
        } else {
            // Equal rank: make ra the root and increment its rank
            self.parent[rb.0 as usize] = ra.0;
            self.rank[ra.0 as usize] += 1;
            ra
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_set_returns_unique_ids() {
        let mut uf = UnionFind::new();
        let a = uf.make_set();
        let b = uf.make_set();
        let c = uf.make_set();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn find_fresh_set_returns_self() {
        let mut uf = UnionFind::new();
        let a = uf.make_set();
        let b = uf.make_set();
        assert_eq!(uf.find(a), a);
        assert_eq!(uf.find(b), b);
    }

    #[test]
    fn union_merges_sets() {
        let mut uf = UnionFind::new();
        let a = uf.make_set();
        let b = uf.make_set();
        uf.union(a, b);
        assert_eq!(uf.find(a), uf.find(b));
    }

    #[test]
    fn path_compression_flattens_chain() {
        let mut uf = UnionFind::new();
        // Build a chain: a -> b -> c -> d by forcing rank-equal unions
        let a = uf.make_set();
        let b = uf.make_set();
        let c = uf.make_set();
        let d = uf.make_set();
        // Force a chain by directly manipulating parent (simulating a worst-case)
        // We do unions in a way that might create depth, then call find on the leaf
        // union(a,b) -> root is a (rank 1)
        // union(c,d) -> root is c (rank 1)
        // union(a,c) -> root is a or c (rank 2)
        uf.union(a, b);
        uf.union(c, d);
        uf.union(a, c);
        // Now find(b) should ultimately reach the root and compress the path
        let root = uf.find(a);
        let root_b = uf.find(b);
        let root_c = uf.find(c);
        let root_d = uf.find(d);
        assert_eq!(root, root_b);
        assert_eq!(root, root_c);
        assert_eq!(root, root_d);
        // After find, parent of b must point directly to root (path compressed)
        assert_eq!(uf.parent[b.0 as usize], root.0);
    }

    #[test]
    fn union_by_rank_stays_balanced() {
        let mut uf = UnionFind::new();
        // Create 8 elements; union them pairwise, then pairs of pairs, etc.
        let ids: Vec<ClassId> = (0..8).map(|_| uf.make_set()).collect();
        // Round 1
        uf.union(ids[0], ids[1]);
        uf.union(ids[2], ids[3]);
        uf.union(ids[4], ids[5]);
        uf.union(ids[6], ids[7]);
        // Round 2
        uf.union(ids[0], ids[2]);
        uf.union(ids[4], ids[6]);
        // Round 3
        uf.union(ids[0], ids[4]);
        // All should have the same representative
        let root = uf.find(ids[0]);
        for &id in &ids {
            assert_eq!(uf.find(id), root);
        }
        // Max rank should be 3 (log2(8))
        let max_rank = *uf.rank.iter().max().unwrap();
        assert!(max_rank <= 3);
    }
}
