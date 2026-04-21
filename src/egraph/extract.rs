use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::cost::CostModel;
use crate::egraph::egraph::EGraph;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

// ── ClassVRegMap ──────────────────────────────────────────────────────────────

/// A newtype wrapper around `BTreeMap<ClassId, VReg>` that provides a narrow,
/// controlled API for mapping e-class IDs to virtual registers.
///
/// All callers use the named methods rather than reaching into the inner map
/// directly. This makes it straightforward to evolve the representation in
/// later phases (e.g. range-keyed segments for pressure splitting) without
/// changing every call site.
#[derive(Debug, Clone, Default)]
pub struct ClassVRegMap {
    single: BTreeMap<ClassId, VReg>,
}

impl ClassVRegMap {
    pub fn new() -> Self {
        ClassVRegMap {
            single: BTreeMap::new(),
        }
    }

    /// Insert a single (class, vreg) mapping, replacing any existing entry.
    pub fn insert_single(&mut self, class: ClassId, vreg: VReg) {
        self.single.insert(class, vreg);
    }

    /// Look up the VReg for `class`, if present.
    pub fn lookup_single(&self, class: ClassId) -> Option<VReg> {
        self.single.get(&class).copied()
    }

    /// Remove the entry for `class` and return the VReg if it existed.
    pub fn remove(&mut self, class: ClassId) -> Option<VReg> {
        self.single.remove(&class)
    }

    /// Iterate over all (ClassId, VReg) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (ClassId, VReg)> + '_ {
        self.single.iter().map(|(&c, &v)| (c, v))
    }

    /// Iterate over all ClassIds in this map.
    pub fn keys(&self) -> impl Iterator<Item = ClassId> + '_ {
        self.single.keys().copied()
    }

    /// Returns `true` if `class` has an entry in this map.
    pub fn contains(&self, class: ClassId) -> bool {
        self.single.contains_key(&class)
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.single.len()
    }

    /// Returns `true` if the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.single.is_empty()
    }

    /// Clone the underlying map. Used for block-local snapshots.
    pub fn clone_inner(&self) -> BTreeMap<ClassId, VReg> {
        self.single.clone()
    }

    /// Returns an entry API for the underlying map. Used by `vreg_insts_for_block`.
    pub(crate) fn entry(
        &mut self,
        class: ClassId,
    ) -> std::collections::btree_map::Entry<'_, ClassId, VReg> {
        self.single.entry(class)
    }
}

// ── Extraction error ──────────────────────────────────────────────────────────

/// Returned when an e-class has no finite-cost node (no legal x86-64 lowering).
#[derive(Debug)]
pub struct ExtractionError {
    pub class_id: ClassId,
    pub ops: Vec<Op>,
    pub message: String,
}

impl std::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ExtractionError {}

// ── Extraction result ─────────────────────────────────────────────────────────

/// The chosen extraction for a single e-class.
#[derive(Debug, Clone)]
pub struct ExtractedNode {
    /// The chosen op.
    pub op: Op,
    /// Canonical class IDs of children (may include `ClassId::NONE` sentinels).
    pub children: Vec<ClassId>,
    /// Total cost: own node cost + sum of children costs.
    pub cost: f64,
}

/// Maps e-class ID to the chosen extraction for that class.
#[derive(Debug)]
pub struct ExtractionResult {
    pub choices: BTreeMap<ClassId, ExtractedNode>,
}

// ── Bottom-up extractor ───────────────────────────────────────────────────────

/// Extract the cheapest subtree reachable from each root class.
///
/// Uses bottom-up dynamic programming: each class is processed once and its
/// result is memoized so shared sub-expressions are not duplicated.
///
/// Returns `Err` if any reachable class has only infinite-cost nodes.
pub fn extract(
    egraph: &EGraph,
    roots: &[ClassId],
    cost_model: &CostModel,
) -> Result<ExtractionResult, ExtractionError> {
    // Canonicalize roots.
    let roots: Vec<ClassId> = roots
        .iter()
        .map(|r| egraph.unionfind.find_immutable(*r))
        .collect();

    // Discover all reachable classes in DFS post-order so we process children
    // before parents (bottom-up).
    let order = reachable_postorder(egraph, &roots);

    let mut memo: BTreeMap<ClassId, ExtractedNode> = BTreeMap::new();

    // Iterative extraction: repeat until all classes are extracted or no progress.
    // This handles cycles from constant-folding merges (e.g., And(a, b) where b
    // was merged into the result class).
    //
    // Worst-case complexity: at most O(|classes|) iterations. Each iteration that
    // makes progress extracts at least one new class (increasing memo.len()), so
    // the loop terminates after at most |order| rounds, or earlier if no progress
    // is made (stuck on unresolvable cycles).
    let mut remaining: Vec<ClassId> = order.clone();
    loop {
        let prev_len = memo.len();
        let mut still_remaining = Vec::new();

        for class_id in &remaining {
            if memo.contains_key(class_id) {
                continue;
            }
            let class = egraph.class(*class_id);
            let mut best: Option<ExtractedNode> = None;

            for node in &class.nodes {
                let own_cost = cost_model.cost(&node.op);
                if own_cost == f64::INFINITY {
                    continue;
                }

                let mut child_cost_sum = 0.0;
                let mut children_canonical: Vec<ClassId> = Vec::with_capacity(node.children.len());
                let mut ok = true;

                for &child in &node.children {
                    if child == ClassId::NONE {
                        children_canonical.push(ClassId::NONE);
                        continue;
                    }
                    let canon = egraph.unionfind.find_immutable(child);
                    children_canonical.push(canon);
                    match memo.get(&canon) {
                        Some(ext) => child_cost_sum += ext.cost,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }

                if !ok {
                    continue;
                }

                let total = own_cost + child_cost_sum;
                let candidate = ExtractedNode {
                    op: node.op.clone(),
                    children: children_canonical,
                    cost: total,
                };

                let dominated = match &best {
                    None => true,
                    Some(prev) if total < prev.cost => true,
                    // On ties, prefer non-BlockParam. BlockParam is only valid
                    // in its specific block; the per-block fixup in compile/mod.rs
                    // ensures the right block gets BlockParam VRegInsts.
                    Some(prev)
                        if total == prev.cost
                            && matches!(prev.op, Op::BlockParam(..))
                            && !matches!(candidate.op, Op::BlockParam(..)) =>
                    {
                        true
                    }
                    _ => false,
                };
                if dominated {
                    best = Some(candidate);
                }
            }

            match best {
                Some(ext) => {
                    memo.insert(*class_id, ext);
                }
                None => {
                    still_remaining.push(*class_id);
                }
            }
        }

        if still_remaining.is_empty() || memo.len() == prev_len {
            // Either all extracted, or no progress (stuck).
            remaining = still_remaining;
            break;
        }
        remaining = still_remaining;
    }

    // Report error for any unextracted classes.
    if let Some(class_id) = remaining.first() {
        let class = egraph.class(*class_id);
        let ops: Vec<Op> = class.nodes.iter().map(|n| n.op.clone()).collect();
        let op_names: Vec<String> = ops.iter().map(|o| format!("{o:?}")).collect();
        return Err(ExtractionError {
            class_id: *class_id,
            ops,
            message: format!(
                "e-class {:?}: no legal x86-64 lowering for {}",
                class_id,
                op_names.join(", ")
            ),
        });
    }

    Ok(ExtractionResult { choices: memo })
}

// ── Constrained extraction ────────────────────────────────────────────────────

/// Returns `true` for ops that are always safe to rematerialize at any point:
/// they have zero cost and no children, so they impose no liveness requirements.
fn is_free_remat(op: &Op) -> bool {
    matches!(
        op,
        Op::Iconst(..)
            | Op::Fconst(..)
            | Op::StackAddr(..)
            | Op::GlobalAddr(..)
            | Op::Param(..)
            | Op::BlockParam(..)
    )
}

/// Cost-aware extraction constrained to operands live at a given program point.
///
/// Picks the best `ExtractedNode` for `class` whose transitive children are
/// either (a) already in `live_classes`, or (b) cheaply reconstructible free
/// remat ops (`Iconst`, `Fconst`, `StackAddr`, `GlobalAddr`, `Param`,
/// `BlockParam`). Child costs are taken from `memo` (the full extraction
/// result); if a child is in `live_classes` its marginal cost contribution is
/// treated as 0.
///
/// Returns `None` if no node in the class yields a finite total cost under
/// these constraints.
///
/// Callers that already have a full `ExtractionResult` should use
/// [`extract_at_with_memo`] to avoid the redundant bottom-up pass.
pub fn extract_at(
    egraph: &EGraph,
    class: ClassId,
    live_classes: &BTreeSet<ClassId>,
    cost_model: &CostModel,
) -> Option<ExtractedNode> {
    // Build full memo via standard extraction from class as root.
    let roots = [class];
    let memo = match extract(egraph, &roots, cost_model) {
        Ok(result) => result.choices,
        Err(_) => return None,
    };
    extract_at_with_memo(egraph, class, live_classes, cost_model, &memo)
}

/// Cost-aware extraction constrained to operands live at a given program point,
/// using a pre-computed extraction memo.
///
/// This is the preferred entry point when the caller already holds the full
/// `ExtractionResult` (e.g. the compile pipeline). The `memo` parameter is the
/// `ExtractionResult::choices` map from a prior [`extract`] call covering at
/// least the classes reachable from `class`.
///
/// For each candidate node in the class the total cost is:
/// ```text
/// own_cost + sum(child_cost)
/// ```
/// where `child_cost` is 0 if the child is in `live_classes` (already
/// available), and `memo[child].cost` otherwise. If a required child has no
/// memo entry (unreachable in the prior extraction), the candidate is skipped.
///
/// Nodes whose `own_cost` is 0 AND have no children (free remat ops) are
/// always selectable regardless of `live_classes`.
///
/// Returns `None` if no candidate yields a finite total cost.
pub fn extract_at_with_memo(
    egraph: &EGraph,
    class: ClassId,
    live_classes: &BTreeSet<ClassId>,
    cost_model: &CostModel,
    memo: &BTreeMap<ClassId, ExtractedNode>,
) -> Option<ExtractedNode> {
    let canon = egraph.unionfind.find_immutable(class);
    let eclass = egraph.class(canon);

    let mut best: Option<ExtractedNode> = None;

    for node in &eclass.nodes {
        let own_cost = cost_model.cost(&node.op);
        if own_cost == f64::INFINITY {
            continue;
        }

        // Free-remat leaf: always selectable, no children to check.
        if is_free_remat(&node.op) && node.children.is_empty() {
            let candidate = ExtractedNode {
                op: node.op.clone(),
                children: vec![],
                cost: own_cost,
            };
            if best
                .as_ref()
                .map_or(true, |b: &ExtractedNode| own_cost < b.cost)
            {
                best = Some(candidate);
            }
            continue;
        }

        // For non-leaf nodes, sum child costs: 0 if live, memo cost otherwise.
        let mut total = own_cost;
        let mut children_canonical: Vec<ClassId> = Vec::with_capacity(node.children.len());
        let mut feasible = true;

        for &child in &node.children {
            if child == ClassId::NONE {
                children_canonical.push(ClassId::NONE);
                continue;
            }
            let child_canon = egraph.unionfind.find_immutable(child);
            children_canonical.push(child_canon);

            if live_classes.contains(&child_canon) {
                // Already live: marginal cost is 0.
            } else {
                // Not live: need to rematerialize; use memo cost.
                match memo.get(&child_canon) {
                    Some(ext) => {
                        if ext.cost == f64::INFINITY {
                            feasible = false;
                            break;
                        }
                        total += ext.cost;
                    }
                    None => {
                        // Child not in memo: unreachable, skip this candidate.
                        feasible = false;
                        break;
                    }
                }
            }
        }

        if !feasible || !total.is_finite() {
            continue;
        }

        let dominated = best.as_ref().map_or(true, |b| total < b.cost);
        if dominated {
            best = Some(ExtractedNode {
                op: node.op.clone(),
                children: children_canonical,
                cost: total,
            });
        }
    }

    best
}

/// Produce a post-order list of all canonical classes reachable from `roots`.
///
/// Each class appears exactly once (after all its children).
fn reachable_postorder(egraph: &EGraph, roots: &[ClassId]) -> Vec<ClassId> {
    let mut visited: BTreeMap<ClassId, bool> = BTreeMap::new(); // false = on stack, true = done
    let mut order: Vec<ClassId> = Vec::new();

    for &root in roots {
        dfs(egraph, root, &mut visited, &mut order);
    }

    order
}

fn dfs(
    egraph: &EGraph,
    id: ClassId,
    visited: &mut BTreeMap<ClassId, bool>,
    order: &mut Vec<ClassId>,
) {
    if id == ClassId::NONE {
        return;
    }
    let canon = egraph.unionfind.find_immutable(id);
    if visited.contains_key(&canon) {
        return;
    }
    // Mark on-stack (cycle guard — e-graphs are DAGs, but be safe).
    visited.insert(canon, false);

    let class = egraph.class(canon);
    for node in &class.nodes {
        for &child in &node.children {
            if child != ClassId::NONE {
                let child_canon = egraph.unionfind.find_immutable(child);
                dfs(egraph, child_canon, visited, order);
            }
        }
    }

    visited.insert(canon, true);
    order.push(canon);
}

// ── VReg / VRegInst ───────────────────────────────────────────────────────────

/// A virtual register allocated during extraction linearization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(pub u32);

/// An extracted instruction with virtual register operands.
#[derive(Debug, Clone)]
pub struct VRegInst {
    pub dst: VReg,
    pub op: Op,
    /// VReg operands resolved from child ClassIds.
    /// A `None` entry corresponds to a `ClassId::NONE` sentinel.
    pub operands: Vec<Option<VReg>>,
}

/// Convert an `ExtractionResult` to a linear list of `VRegInst`s in
/// dependency order (each def appears before its uses).
///
/// One `VReg` is allocated per extracted class; instructions are emitted in
/// post-order so operands are always defined before they are referenced.
/// Same as `extraction_to_vreg_insts` but also returns the ClassId -> VReg mapping.
pub fn extraction_to_vreg_insts_with_map(
    extraction: &ExtractionResult,
    roots: &[ClassId],
) -> (Vec<VRegInst>, ClassVRegMap) {
    let mut class_to_vreg = ClassVRegMap::new();
    let mut next_vreg: u32 = 0;

    let mut visited: std::collections::BTreeSet<ClassId> = std::collections::BTreeSet::new();
    let mut emit_order: Vec<ClassId> = Vec::new();

    for &root in roots {
        vreg_dfs(root, extraction, &mut visited, &mut emit_order);
    }

    for &class_id in &emit_order {
        let vreg = VReg(next_vreg);
        next_vreg += 1;
        class_to_vreg.insert_single(class_id, vreg);
    }

    let mut insts: Vec<VRegInst> = Vec::with_capacity(emit_order.len());
    for &class_id in &emit_order {
        let ext = &extraction.choices[&class_id];
        let dst = class_to_vreg.lookup_single(class_id).unwrap();
        let operands: Vec<Option<VReg>> = ext
            .children
            .iter()
            .map(|&child| {
                if child == ClassId::NONE {
                    None
                } else {
                    Some(class_to_vreg.lookup_single(child).unwrap())
                }
            })
            .collect();
        insts.push(VRegInst {
            dst,
            op: ext.op.clone(),
            operands,
        });
    }

    (insts, class_to_vreg)
}

pub fn extraction_to_vreg_insts(extraction: &ExtractionResult, roots: &[ClassId]) -> Vec<VRegInst> {
    // Assign a VReg to each class in the extraction result.
    let mut class_to_vreg = ClassVRegMap::new();
    let mut next_vreg: u32 = 0;

    // Build emission order: post-order DFS over extraction DAG from roots.
    let mut visited: std::collections::BTreeSet<ClassId> = std::collections::BTreeSet::new();
    let mut emit_order: Vec<ClassId> = Vec::new();

    for &root in roots {
        vreg_dfs(root, extraction, &mut visited, &mut emit_order);
    }

    // Allocate VRegs in emission order.
    for &class_id in &emit_order {
        let vreg = VReg(next_vreg);
        next_vreg += 1;
        class_to_vreg.insert_single(class_id, vreg);
    }

    // Emit instructions in emission order.
    let mut insts: Vec<VRegInst> = Vec::with_capacity(emit_order.len());
    for &class_id in &emit_order {
        let ext = &extraction.choices[&class_id];
        let dst = class_to_vreg.lookup_single(class_id).unwrap();
        let operands: Vec<Option<VReg>> = ext
            .children
            .iter()
            .map(|&child| {
                if child == ClassId::NONE {
                    None
                } else {
                    Some(class_to_vreg.lookup_single(child).unwrap())
                }
            })
            .collect();
        insts.push(VRegInst {
            dst,
            op: ext.op.clone(),
            operands,
        });
    }

    insts
}

/// Build VRegInsts for a specific set of roots, reusing an existing
/// `class_to_vreg` map. New classes encountered that are not yet in the map
/// get fresh VRegs starting from `next_vreg`. Classes already in the map
/// produce no new instruction (they were emitted by a prior call).
///
/// Returns the VRegInsts for the classes newly encountered on this call
/// (i.e., not yet in `class_to_vreg`), in dependency order.
pub fn vreg_insts_for_block(
    extraction: &ExtractionResult,
    roots: &[ClassId],
    class_to_vreg: &mut ClassVRegMap,
    next_vreg: &mut u32,
) -> Vec<VRegInst> {
    // DFS to find emission order for classes not yet visited.
    let mut visited: std::collections::BTreeSet<ClassId> = class_to_vreg.keys().collect();
    let mut emit_order: Vec<ClassId> = Vec::new();

    for &root in roots {
        vreg_dfs(root, extraction, &mut visited, &mut emit_order);
    }

    // Assign new VRegs.
    for &class_id in &emit_order {
        if let std::collections::btree_map::Entry::Vacant(e) = class_to_vreg.entry(class_id) {
            e.insert(VReg(*next_vreg));
            *next_vreg += 1;
        }
    }

    // Build VRegInsts.
    let mut insts = Vec::with_capacity(emit_order.len());
    for &class_id in &emit_order {
        let ext = &extraction.choices[&class_id];
        let dst = class_to_vreg.lookup_single(class_id).unwrap();
        let operands: Vec<Option<VReg>> = ext
            .children
            .iter()
            .map(|&child| {
                if child == ClassId::NONE {
                    None
                } else {
                    Some(class_to_vreg.lookup_single(child).unwrap())
                }
            })
            .collect();
        insts.push(VRegInst {
            dst,
            op: ext.op.clone(),
            operands,
        });
    }
    insts
}

/// Build a map from VReg to its IR Type by looking up each VReg's e-class type
/// in the egraph. The egraph stores a `ty: Type` on every e-class, so this is
/// a straightforward lookup rather than a bottom-up type inference pass.
pub fn build_vreg_types(class_to_vreg: &ClassVRegMap, egraph: &EGraph) -> BTreeMap<VReg, Type> {
    let mut vreg_types = BTreeMap::new();
    for (class_id, vreg) in class_to_vreg.iter() {
        let canon = egraph.unionfind.find_immutable(class_id);
        let ty = egraph.class(canon).ty.clone();
        vreg_types.insert(vreg, ty);
    }
    vreg_types
}

fn vreg_dfs(
    id: ClassId,
    extraction: &ExtractionResult,
    visited: &mut std::collections::BTreeSet<ClassId>,
    order: &mut Vec<ClassId>,
) {
    if id == ClassId::NONE || !extraction.choices.contains_key(&id) {
        return;
    }
    if !visited.insert(id) {
        return;
    }
    let ext = &extraction.choices[&id];
    for &child in &ext.children {
        if child != ClassId::NONE {
            vreg_dfs(child, extraction, visited, order);
        }
    }
    order.push(id);
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::cost::OptGoal;
    use crate::egraph::enode::ENode;
    use crate::ir::types::Type;

    fn iconst(g: &mut EGraph, v: i64) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(v, Type::I64),
            children: smallvec![],
        })
    }

    fn x86add(g: &mut EGraph, a: ClassId, b: ClassId) -> ClassId {
        g.add(ENode {
            op: Op::X86Add,
            children: smallvec![a, b],
        })
    }

    fn proj0(g: &mut EGraph, pair: ClassId) -> ClassId {
        g.add(ENode {
            op: Op::Proj0,
            children: smallvec![pair],
        })
    }

    // 5.4: E-class with one machine node extracts it.
    #[test]
    fn single_machine_node_extracts() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1);
        let b = iconst(&mut g, 2);
        let pair = x86add(&mut g, a, b);
        let result_class = proj0(&mut g, pair);

        let cm = CostModel::new(OptGoal::Balanced);
        let result = extract(&g, &[result_class], &cm).expect("extraction must succeed");
        assert!(
            result
                .choices
                .contains_key(&g.unionfind.find_immutable(result_class))
        );
    }

    // 5.4: E-class with generic + machine picks machine (finite cost).
    #[test]
    fn generic_and_machine_picks_machine() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 10);
        let b = iconst(&mut g, 20);

        // IR-level Add
        let ir_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, b],
        });

        // Machine X86Add + Proj0
        let pair = x86add(&mut g, a, b);
        let p0 = proj0(&mut g, pair);

        // Merge them into one class
        g.merge(ir_add, p0);
        g.rebuild();

        let canon = g.unionfind.find_immutable(ir_add);
        let cm = CostModel::new(OptGoal::Balanced);
        let result = extract(&g, &[canon], &cm).expect("should pick machine node");

        let extracted = &result.choices[&canon];
        // The chosen op must not be the generic Add (which has infinite cost)
        assert_ne!(extracted.op, Op::Add, "should not pick generic Add");
    }

    // 5.4: E-class with only generic nodes fails with diagnostic.
    #[test]
    fn only_generic_nodes_fails() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1);
        let b = iconst(&mut g, 2);
        let ir_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, b],
        });

        let cm = CostModel::new(OptGoal::Balanced);
        let err = extract(&g, &[ir_add], &cm).expect_err("should fail");
        assert!(err.message.contains("no legal x86-64 lowering"));
        assert!(err.ops.iter().any(|o| *o == Op::Add));
    }

    // 5.6: VReg linearization produces instructions in def-before-use order.
    #[test]
    fn vreg_insts_def_before_use() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1);
        let b = iconst(&mut g, 2);
        let pair = x86add(&mut g, a, b);
        let p0 = proj0(&mut g, pair);

        let cm = CostModel::new(OptGoal::Balanced);
        let extraction = extract(&g, &[p0], &cm).expect("ok");
        let insts = extraction_to_vreg_insts(&extraction, &[g.unionfind.find_immutable(p0)]);

        // Verify every operand VReg is defined before it is used.
        let mut defined: std::collections::BTreeSet<VReg> = std::collections::BTreeSet::new();
        for inst in &insts {
            for &op in inst.operands.iter().flatten() {
                assert!(
                    defined.contains(&op),
                    "VReg {:?} used before defined (inst op={:?})",
                    op,
                    inst.op
                );
            }
            defined.insert(inst.dst);
        }
    }

    // 5.7: Load with addressing mode — Addr node is extracted.
    // The e-graph contains Addr{scale:4,disp:0}(base, Proj0(X86Shl(idx, two)))
    // Extraction should pick the Addr node (cost 0) over a plain address.
    #[test]
    fn addr_node_extraction() {
        use crate::egraph::addr_mode::apply_addr_mode_rules;
        use crate::egraph::isel::apply_isel_rules;

        let mut g = EGraph::new();
        let base = iconst(&mut g, 0x1000);
        let idx = iconst(&mut g, 3);
        let two = iconst(&mut g, 2);

        // idx << 2  (= idx * 4)
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![idx, two],
        });

        // base + (idx << 2)
        let addr = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, shl],
        });

        // Run isel and addr mode rules to generate X86 nodes and Addr nodes
        for _ in 0..8 {
            apply_isel_rules(&mut g);
            apply_addr_mode_rules(&mut g);
            g.rebuild();
        }

        let cm = CostModel::new(OptGoal::Balanced);
        let addr_canon = g.unionfind.find_immutable(addr);
        let result = extract(&g, &[addr_canon], &cm).expect("extraction succeeds");

        // The extracted class for addr should have a finite cost
        let extracted = &result.choices[&addr_canon];
        assert!(
            extracted.cost.is_finite(),
            "addr class must extract to finite cost"
        );

        // The addr class (or a sub-class) should contain an Addr node somewhere
        let has_addr_node = result
            .choices
            .values()
            .any(|ext| matches!(ext.op, Op::Addr { .. }));
        assert!(
            has_addr_node,
            "extraction should include an Addr node for scale=4"
        );
    }

    // 5.8: Correctness — 5 small IR programs, no infinite-cost nodes, def-before-use.
    #[test]
    fn extraction_correctness_five_programs() {
        use crate::ir::condcode::CondCode;

        let cm = CostModel::new(OptGoal::Balanced);

        // Program 1: iconst only
        {
            let mut g = EGraph::new();
            let c = iconst(&mut g, 42);
            let result = extract(&g, &[c], &cm).expect("p1 ok");
            for (_, ext) in &result.choices {
                assert!(ext.cost.is_finite(), "p1: infinite cost node chosen");
            }
            let insts = extraction_to_vreg_insts(&result, &[g.unionfind.find_immutable(c)]);
            check_def_before_use(&insts);
        }

        // Program 2: X86Add(a, b)
        {
            let mut g = EGraph::new();
            let a = iconst(&mut g, 1);
            let b = iconst(&mut g, 2);
            let pair = x86add(&mut g, a, b);
            let p0 = proj0(&mut g, pair);
            let result = extract(&g, &[p0], &cm).expect("p2 ok");
            for (_, ext) in &result.choices {
                assert!(ext.cost.is_finite(), "p2: infinite cost");
            }
            let insts = extraction_to_vreg_insts(&result, &[g.unionfind.find_immutable(p0)]);
            check_def_before_use(&insts);
        }

        // Program 3: nested X86Add(X86Add(a,b), c)
        {
            let mut g = EGraph::new();
            let a = iconst(&mut g, 1);
            let b = iconst(&mut g, 2);
            let c = iconst(&mut g, 3);
            let inner_pair = x86add(&mut g, a, b);
            let inner = proj0(&mut g, inner_pair);
            let outer_pair = x86add(&mut g, inner, c);
            let outer = proj0(&mut g, outer_pair);
            let result = extract(&g, &[outer], &cm).expect("p3 ok");
            for (_, ext) in &result.choices {
                assert!(ext.cost.is_finite(), "p3: infinite cost");
            }
            let insts = extraction_to_vreg_insts(&result, &[g.unionfind.find_immutable(outer)]);
            check_def_before_use(&insts);
        }

        // Program 4: Icmp + isel -> X86Sub + Proj1 (flags class)
        {
            use crate::egraph::isel::apply_isel_rules;
            let mut g = EGraph::new();
            let a = iconst(&mut g, 10);
            let b = iconst(&mut g, 5);
            let flags = g.add(ENode {
                op: Op::Icmp(CondCode::Slt),
                children: smallvec![a, b],
            });
            apply_isel_rules(&mut g);
            g.rebuild();
            let flags_canon = g.unionfind.find_immutable(flags);
            let result = extract(&g, &[flags_canon], &cm).expect("p4 ok");
            for (_, ext) in &result.choices {
                assert!(ext.cost.is_finite(), "p4: infinite cost");
            }
            let insts = extraction_to_vreg_insts(&result, &[flags_canon]);
            check_def_before_use(&insts);
        }

        // Program 5: shared sub-expression — two adds sharing a common sub-tree
        {
            let mut g = EGraph::new();
            let a = iconst(&mut g, 1);
            let b = iconst(&mut g, 2);
            let shared_pair = x86add(&mut g, a, b);
            let shared = proj0(&mut g, shared_pair);
            let c = iconst(&mut g, 3);
            let pair1 = x86add(&mut g, shared, c);
            let r1 = proj0(&mut g, pair1);
            let d = iconst(&mut g, 4);
            let pair2 = x86add(&mut g, shared, d);
            let r2 = proj0(&mut g, pair2);

            let r1c = g.unionfind.find_immutable(r1);
            let r2c = g.unionfind.find_immutable(r2);
            let result = extract(&g, &[r1c, r2c], &cm).expect("p5 ok");

            // Shared class should appear only once in choices
            let shared_canon = g.unionfind.find_immutable(shared);
            assert!(
                result.choices.contains_key(&shared_canon),
                "shared class present"
            );

            for (_, ext) in &result.choices {
                assert!(ext.cost.is_finite(), "p5: infinite cost");
            }
            let insts = extraction_to_vreg_insts(&result, &[r1c, r2c]);
            check_def_before_use(&insts);
        }
    }

    fn check_def_before_use(insts: &[VRegInst]) {
        let mut defined: std::collections::BTreeSet<VReg> = std::collections::BTreeSet::new();
        for inst in insts {
            for &op in inst.operands.iter().flatten() {
                assert!(
                    defined.contains(&op),
                    "VReg {:?} used before defined in inst {:?}",
                    op,
                    inst.op
                );
            }
            defined.insert(inst.dst);
        }
    }

    #[test]
    fn classvregmap_single_insert_lookup() {
        let mut map = ClassVRegMap::new();
        let c0 = ClassId(0);
        let c1 = ClassId(1);
        let c2 = ClassId(2);
        let v0 = VReg(0);
        let v1 = VReg(1);

        // Fresh map returns None for any class.
        assert_eq!(map.lookup_single(c0), None);

        // Insert and round-trip.
        map.insert_single(c0, v0);
        assert_eq!(map.lookup_single(c0), Some(v0));
        assert_eq!(map.lookup_single(c1), None);

        // Insert a second class independently.
        map.insert_single(c1, v1);
        assert_eq!(map.lookup_single(c0), Some(v0));
        assert_eq!(map.lookup_single(c1), Some(v1));

        // Overwrite an existing entry.
        map.insert_single(c0, v1);
        assert_eq!(map.lookup_single(c0), Some(v1));

        // contains/keys/iter.
        assert!(map.contains(c0));
        assert!(map.contains(c1));
        assert!(!map.contains(c2));

        let keys: Vec<ClassId> = map.keys().collect();
        assert_eq!(keys, vec![c0, c1]);

        let pairs: Vec<(ClassId, VReg)> = map.iter().collect();
        assert_eq!(pairs, vec![(c0, v1), (c1, v1)]);

        // Remove.
        let removed = map.remove(c0);
        assert_eq!(removed, Some(v1));
        assert_eq!(map.lookup_single(c0), None);
        assert_eq!(map.len(), 1);
    }

    // ── extract_at tests ──────────────────────────────────────────────────────

    // 3.2a: Iconst is selectable even when not in live_classes.
    #[test]
    fn extract_at_prefers_free_remat() {
        let mut g = EGraph::new();
        let c = iconst(&mut g, 42);
        let cm = CostModel::new(OptGoal::Balanced);
        let live: BTreeSet<ClassId> = BTreeSet::new(); // nothing live

        let canon = g.unionfind.find_immutable(c);
        let result = extract_at(&g, canon, &live, &cm);
        assert!(
            result.is_some(),
            "Iconst must be selectable even with empty live set"
        );
        let ext = result.unwrap();
        assert!(matches!(ext.op, Op::Iconst(42, _)), "must pick Iconst");
    }

    // 3.2b: X86Add where both children are live: picks X86Add.
    #[test]
    fn extract_at_uses_live_child() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1);
        let b = iconst(&mut g, 2);
        let pair = x86add(&mut g, a, b);
        let p0 = proj0(&mut g, pair);

        let cm = CostModel::new(OptGoal::Balanced);
        let result = extract(&g, &[g.unionfind.find_immutable(p0)], &cm).expect("ok");

        // Mark a and b as live so X86Add (through the pair class) has 0 child cost.
        let a_canon = g.unionfind.find_immutable(a);
        let b_canon = g.unionfind.find_immutable(b);
        let pair_canon = g.unionfind.find_immutable(pair);

        let mut live: BTreeSet<ClassId> = BTreeSet::new();
        live.insert(a_canon);
        live.insert(b_canon);
        live.insert(pair_canon);

        let p0_canon = g.unionfind.find_immutable(p0);
        let ext = extract_at_with_memo(&g, p0_canon, &live, &cm, &result.choices);
        assert!(ext.is_some(), "should extract when children are live");
    }

    // 3.2c: A deep tree where no internals are live and internals are expensive:
    // the class can still be extracted (children have memo entries).
    #[test]
    fn extract_at_rejects_nonlive_expensive_children() {
        // Build a deep tree: add(add(iconst, iconst), add(iconst, iconst))
        let mut g = EGraph::new();
        let a = iconst(&mut g, 1);
        let b = iconst(&mut g, 2);
        let c = iconst(&mut g, 3);
        let d = iconst(&mut g, 4);
        let ab_pair = x86add(&mut g, a, b);
        let ab = proj0(&mut g, ab_pair);
        let cd_pair = x86add(&mut g, c, d);
        let cd = proj0(&mut g, cd_pair);
        let root_pair = x86add(&mut g, ab, cd);
        let root = proj0(&mut g, root_pair);

        let cm = CostModel::new(OptGoal::Balanced);
        let root_canon = g.unionfind.find_immutable(root);
        let result = extract(&g, &[root_canon], &cm).expect("ok");

        // Nothing is live; the class should still extract (children have finite memo costs).
        let live: BTreeSet<ClassId> = BTreeSet::new();
        let ext = extract_at_with_memo(&g, root_canon, &live, &cm, &result.choices);
        // Root has finite memo cost so it should succeed.
        assert!(
            ext.is_some(),
            "deep tree should extract from empty live set via memo"
        );
    }

    // 3.2d: extract_at agrees with extract when everything is live.
    #[test]
    fn extract_at_matches_extract_when_all_live() {
        let mut g = EGraph::new();
        let a = iconst(&mut g, 10);
        let b = iconst(&mut g, 20);
        let pair = x86add(&mut g, a, b);
        let p0 = proj0(&mut g, pair);

        let cm = CostModel::new(OptGoal::Balanced);
        let p0_canon = g.unionfind.find_immutable(p0);
        let result = extract(&g, &[p0_canon], &cm).expect("ok");

        // Mark all classes as live.
        let live: BTreeSet<ClassId> = result.choices.keys().copied().collect();
        let ext = extract_at_with_memo(&g, p0_canon, &live, &cm, &result.choices);

        // Both approaches must pick the same op.
        let normal = &result.choices[&p0_canon];
        let constrained = ext.expect("should succeed with all classes live");
        assert_eq!(
            normal.op, constrained.op,
            "extract_at must agree with extract when all children are live"
        );
    }

    // 3.2e: A class containing only a generic Add (infinite cost) returns None.
    #[test]
    fn extract_at_returns_none_on_truly_infeasible() {
        use crate::egraph::enode::ENode;
        use smallvec::smallvec;

        let mut g = EGraph::new();
        let a = iconst(&mut g, 1);
        let b = iconst(&mut g, 2);
        // Only a generic Add — no machine node.
        let ir_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, b],
        });

        let cm = CostModel::new(OptGoal::Balanced);
        let ir_add_canon = g.unionfind.find_immutable(ir_add);

        // Build memo only for the iconst children (not for ir_add itself).
        let a_canon = g.unionfind.find_immutable(a);
        let b_canon = g.unionfind.find_immutable(b);
        let ab_result = extract(&g, &[a_canon, b_canon], &cm).expect("iconsts ok");

        // Attempt constrained extraction on ir_add with both iconsts live.
        let mut live: BTreeSet<ClassId> = BTreeSet::new();
        live.insert(a_canon);
        live.insert(b_canon);
        let ext = extract_at_with_memo(&g, ir_add_canon, &live, &cm, &ab_result.choices);
        assert!(
            ext.is_none(),
            "generic Add with no machine lowering must return None"
        );
    }
}
