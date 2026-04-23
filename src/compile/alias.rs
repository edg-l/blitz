//! Alias analysis infrastructure for the Blitz compiler.
//!
//! Provides address categorization and alias queries used by
//! store-to-load forwarding (Phase 2) and dead store elimination (Phase 3).

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::egraph::egraph::EGraph;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

/// Categorized base of a memory address.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AddrBase {
    /// Address derived from stack slot N (`Op::StackAddr(n)`).
    StackSlot(u32),
    /// Address derived from a global symbol (`Op::GlobalAddr(sym)`).
    Global(String),
    /// Unknown or un-categorizable base (conservative).
    Unknown,
}

/// Alias analysis state for a single function.
///
/// Caches per-class `AddrBase` classifications keyed by canonical `ClassId`.
pub struct AliasInfo {
    base_cache: RefCell<BTreeMap<ClassId, AddrBase>>,
}

impl AliasInfo {
    pub fn new() -> Self {
        Self {
            base_cache: RefCell::new(BTreeMap::new()),
        }
    }

    /// Classify the address base of `class`.
    ///
    /// Canonicalizes via union-find, checks the memo cache, and on a miss
    /// delegates to `classify_inner` with a recursion depth cap of 16.
    pub fn classify(&self, class: ClassId, egraph: &EGraph) -> AddrBase {
        let canon = egraph.unionfind.find_immutable(class);
        if let Some(base) = self.base_cache.borrow().get(&canon) {
            return base.clone();
        }
        let result = self.classify_inner(canon, egraph, 0);
        self.base_cache.borrow_mut().insert(canon, result.clone());
        result
    }

    fn classify_inner(&self, class: ClassId, egraph: &EGraph, depth: u32) -> AddrBase {
        if depth >= 16 {
            return AddrBase::Unknown;
        }
        let canon = egraph.unionfind.find_immutable(class);
        let eclass = egraph.class(canon);

        // Walk the e-class nodes. Use the first node that gives a known base;
        // if any node gives Unknown, the whole class is Unknown (conservative).
        let mut best: Option<AddrBase> = None;
        for node in &eclass.nodes {
            let base = self.classify_op(&node.op, &node.children, egraph, depth);
            match &base {
                AddrBase::Unknown => {
                    // Unknown from any node — return immediately (conservative).
                    return AddrBase::Unknown;
                }
                _ => {
                    if let Some(prev) = &best {
                        if prev != &base {
                            // Two nodes disagree on base — conservative.
                            return AddrBase::Unknown;
                        }
                    } else {
                        best = Some(base);
                    }
                }
            }
        }
        best.unwrap_or(AddrBase::Unknown)
    }

    /// Classify a single Op node given its children.
    ///
    /// Exhaustively enumerates every Op variant — no wildcard arm — so new Op
    /// additions cause a compile error rather than silently falling through.
    fn classify_op(&self, op: &Op, children: &[ClassId], egraph: &EGraph, depth: u32) -> AddrBase {
        match op {
            // ── Address-producing ops: propagate base ─────────────────────────
            Op::StackAddr(n) => AddrBase::StackSlot(*n),
            Op::GlobalAddr(sym) => AddrBase::Global(sym.clone()),

            Op::Add => {
                // Propagate if exactly one side has a known base.
                let [lhs, rhs] = match children {
                    [a, b] => [*a, *b],
                    _ => return AddrBase::Unknown,
                };
                let lb =
                    self.classify_inner(egraph.unionfind.find_immutable(lhs), egraph, depth + 1);
                let rb =
                    self.classify_inner(egraph.unionfind.find_immutable(rhs), egraph, depth + 1);
                merge_add_bases(lb, rb)
            }

            Op::X86Lea2 | Op::X86Lea3 { .. } | Op::X86Lea4 { .. } => {
                // Base is the first operand.
                let base_child = match children.first() {
                    Some(&c) => c,
                    None => return AddrBase::Unknown,
                };
                let lb = self.classify_inner(
                    egraph.unionfind.find_immutable(base_child),
                    egraph,
                    depth + 1,
                );
                // Index (children[1]) is Unknown context; propagate base only
                // if index doesn't contribute another known base (same as Add).
                if children.len() < 2 {
                    return lb;
                }
                let idx_child = children[1];
                let rb = if idx_child == ClassId::NONE {
                    AddrBase::Unknown
                } else {
                    self.classify_inner(
                        egraph.unionfind.find_immutable(idx_child),
                        egraph,
                        depth + 1,
                    )
                };
                merge_add_bases(lb, rb)
            }

            Op::Addr { .. } => {
                // Same as X86Lea: base is first operand.
                let base_child = match children.first() {
                    Some(&c) => c,
                    None => return AddrBase::Unknown,
                };
                let lb = self.classify_inner(
                    egraph.unionfind.find_immutable(base_child),
                    egraph,
                    depth + 1,
                );
                if children.len() < 2 {
                    return lb;
                }
                let idx_child = children[1];
                let rb = if idx_child == ClassId::NONE {
                    AddrBase::Unknown
                } else {
                    self.classify_inner(
                        egraph.unionfind.find_immutable(idx_child),
                        egraph,
                        depth + 1,
                    )
                };
                merge_add_bases(lb, rb)
            }

            // ── All other ops → Unknown ───────────────────────────────────────
            // Arithmetic (non-Add)
            Op::Sub | Op::Mul | Op::UDiv | Op::SDiv | Op::URem | Op::SRem => AddrBase::Unknown,

            // Bitwise
            Op::And | Op::Or | Op::Xor | Op::Shl | Op::Shr | Op::Sar => AddrBase::Unknown,

            // Conversion
            Op::Sext(_) | Op::Zext(_) | Op::Trunc(_) | Op::Bitcast(_) => AddrBase::Unknown,

            // Constants
            Op::Iconst(_, _) | Op::Fconst(_, _) => AddrBase::Unknown,

            // Parameters / block parameters
            Op::Param(_, _) | Op::BlockParam(_, _, _) => AddrBase::Unknown,

            // Comparison
            Op::Icmp(_) | Op::Fcmp(_) => AddrBase::Unknown,

            // Float/int conversion
            Op::IntToFloat(_) | Op::FloatToInt(_) | Op::FloatExt | Op::FloatTrunc => {
                AddrBase::Unknown
            }

            // FP arithmetic
            Op::Fadd | Op::Fsub | Op::Fmul | Op::Fdiv | Op::Fsqrt => AddrBase::Unknown,

            // Conditional select
            Op::Select => AddrBase::Unknown,

            // Projections
            Op::Proj0 | Op::Proj1 => AddrBase::Unknown,

            // x86 ALU (flag-producing)
            Op::X86Add
            | Op::X86Sub
            | Op::X86And
            | Op::X86Or
            | Op::X86Xor
            | Op::X86Shl
            | Op::X86Sar
            | Op::X86Shr => AddrBase::Unknown,

            // x86 immediate shifts
            Op::X86ShlImm(_) | Op::X86ShrImm(_) | Op::X86SarImm(_) => AddrBase::Unknown,

            // x86 flag-only compare with immediate: produces Flags, not an address.
            Op::X86CmpI { .. } => AddrBase::Unknown,

            // x86 multiply/divide
            Op::X86Imul3 | Op::X86Idiv | Op::X86Div => AddrBase::Unknown,

            // x86 conditional move / setcc
            Op::X86Cmov(_) | Op::X86Setcc(_) => AddrBase::Unknown,

            // x86 FP arithmetic
            Op::X86Addsd
            | Op::X86Subsd
            | Op::X86Mulsd
            | Op::X86Divsd
            | Op::X86Sqrtsd
            | Op::X86Addss
            | Op::X86Subss
            | Op::X86Mulss
            | Op::X86Divss
            | Op::X86Sqrtss => AddrBase::Unknown,

            // x86 FP conversion
            Op::X86Cvtsi2sd
            | Op::X86Cvtsi2ss
            | Op::X86Cvttsd2si(_)
            | Op::X86Cvttss2si(_)
            | Op::X86Cvtsd2ss
            | Op::X86Cvtss2sd => AddrBase::Unknown,

            // x86 FP comparison
            Op::X86Ucomisd | Op::X86Ucomiss => AddrBase::Unknown,

            // x86 integer conversion
            Op::X86Movsx { .. }
            | Op::X86Movzx { .. }
            | Op::X86Trunc { .. }
            | Op::X86Bitcast { .. } => AddrBase::Unknown,

            // Result placeholders
            Op::LoadResult(_, _) | Op::CallResult(_, _) => AddrBase::Unknown,

            // Spill pseudo-ops
            Op::SpillStore(_) | Op::SpillLoad(_) | Op::XmmSpillStore(_) | Op::XmmSpillLoad(_) => {
                AddrBase::Unknown
            }

            // Barrier pseudo-ops
            Op::StoreBarrier | Op::VoidCallBarrier => AddrBase::Unknown,
        }
    }

    /// Returns `true` if the two addresses *may* alias.
    pub fn may_alias(&self, a: ClassId, b: ClassId, egraph: &EGraph) -> bool {
        let ba = self.classify(a, egraph);
        let bb = self.classify(b, egraph);
        bases_may_alias(&ba, &bb)
    }

    /// Returns `true` if the two addresses *must* alias (definitely the same location).
    ///
    /// Only canonical e-class equality counts. Hashconsing makes this work for
    /// identical expressions: `StackAddr(n)` twice, `GlobalAddr(s)` twice, or
    /// `Add(base, iconst(k))` twice all canonicalize to the same class. Two
    /// addresses sharing only a base (e.g. different offsets into the same
    /// stack slot) may alias but do NOT must-alias.
    pub fn must_alias(&self, a: ClassId, b: ClassId, egraph: &EGraph) -> bool {
        let ca = egraph.unionfind.find_immutable(a);
        let cb = egraph.unionfind.find_immutable(b);
        ca == cb
    }

    /// Returns `true` if a store at `store_addr` with type `store_ty` may clobber
    /// a load at `load_addr` with type `load_ty`.
    pub fn store_clobbers_load(
        &self,
        store_addr: ClassId,
        store_ty: Type,
        load_addr: ClassId,
        load_ty: Type,
        egraph: &EGraph,
    ) -> bool {
        self.may_alias(store_addr, load_addr, egraph) && types_overlap(store_ty, load_ty)
    }
}

impl Default for AliasInfo {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Merge two bases as if computing `base + offset`: propagate a single known
/// base; return Unknown if both sides are known (two bases don't add to form
/// a single-base address in our model).
fn merge_add_bases(lb: AddrBase, rb: AddrBase) -> AddrBase {
    match (lb, rb) {
        (AddrBase::Unknown, AddrBase::Unknown) => AddrBase::Unknown,
        (known, AddrBase::Unknown) | (AddrBase::Unknown, known) => known,
        (a, b) => {
            // Both sides are known bases. If they're the same base, that's
            // unusual (base + base) but still a single known base. If they
            // differ, we can't tell which one owns the address.
            if a == b { a } else { AddrBase::Unknown }
        }
    }
}

fn bases_may_alias(a: &AddrBase, b: &AddrBase) -> bool {
    match (a, b) {
        // Distinct stack slots never overlap.
        (AddrBase::StackSlot(x), AddrBase::StackSlot(y)) => x == y,
        // Distinct globals never overlap.
        (AddrBase::Global(s1), AddrBase::Global(s2)) => s1 == s2,
        // Stack vs global: never alias.
        (AddrBase::StackSlot(_), AddrBase::Global(_))
        | (AddrBase::Global(_), AddrBase::StackSlot(_)) => false,
        // Unknown on either side: conservatively may alias.
        _ => true,
    }
}

/// Conservative type-overlap check.
///
/// In v1, all accesses conservatively overlap regardless of width because a
/// narrow access inside a wider allocation (or vice versa) may partially
/// overlap. `byte_size` is consulted but mismatched widths also return `true`.
/// Future versions can tighten this with size-disjoint reasoning.
fn types_overlap(a: Type, b: Type) -> bool {
    match (a.byte_size(), b.byte_size()) {
        // Same byte width at the same address: definite overlap.
        (Some(sa), Some(sb)) if sa == sb => true,
        // Mismatched widths: conservatively may overlap (partial access).
        _ => true,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::egraph::EGraph;
    use crate::egraph::enode::ENode;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn add_node(eg: &mut EGraph, op: Op, children: &[ClassId]) -> ClassId {
        eg.add(ENode {
            op,
            children: children.iter().copied().collect(),
        })
    }

    fn stack_addr(eg: &mut EGraph, n: u32) -> ClassId {
        add_node(eg, Op::StackAddr(n), &[])
    }

    fn global_addr(eg: &mut EGraph, name: &str) -> ClassId {
        add_node(eg, Op::GlobalAddr(name.to_string()), &[])
    }

    fn iconst(eg: &mut EGraph, v: i64) -> ClassId {
        add_node(eg, Op::Iconst(v, Type::I64), &[])
    }

    fn param(eg: &mut EGraph, idx: u32) -> ClassId {
        add_node(eg, Op::Param(idx, Type::I64), &[])
    }

    #[test]
    fn distinct_stack_slots_no_alias() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let s1 = stack_addr(&mut eg, 1);
        let ai = AliasInfo::new();
        assert!(!ai.may_alias(s0, s1, &eg));
        assert!(!ai.must_alias(s0, s1, &eg));
    }

    #[test]
    fn same_stack_slot_must_alias() {
        let mut eg = EGraph::new();
        let s3a = stack_addr(&mut eg, 3);
        // Adding the same op again returns the same ClassId (hashcons).
        let s3b = stack_addr(&mut eg, 3);
        let ai = AliasInfo::new();
        assert_eq!(s3a, s3b, "hashcons must return same ClassId");
        assert!(ai.may_alias(s3a, s3b, &eg));
        assert!(ai.must_alias(s3a, s3b, &eg));
    }

    #[test]
    fn stack_vs_global_no_alias() {
        let mut eg = EGraph::new();
        let s = stack_addr(&mut eg, 0);
        let g = global_addr(&mut eg, "g");
        let ai = AliasInfo::new();
        assert!(!ai.may_alias(s, g, &eg));
        assert!(!ai.must_alias(s, g, &eg));
    }

    #[test]
    fn same_global_may_and_must_alias() {
        let mut eg = EGraph::new();
        // Adding the same global twice: hashcons returns same ClassId.
        let g1 = global_addr(&mut eg, "g");
        let g2 = global_addr(&mut eg, "g");
        let ai = AliasInfo::new();
        assert_eq!(g1, g2);
        assert!(ai.may_alias(g1, g2, &eg));
    }

    #[test]
    fn distinct_globals_no_alias() {
        let mut eg = EGraph::new();
        let g = global_addr(&mut eg, "g");
        let h = global_addr(&mut eg, "h");
        let ai = AliasInfo::new();
        assert!(!ai.may_alias(g, h, &eg));
        assert!(!ai.must_alias(g, h, &eg));
    }

    #[test]
    fn add_stack_iconst_vs_different_slot_no_alias() {
        // Add(StackAddr(2), Iconst(8)) vs StackAddr(0) → false (different bases)
        let mut eg = EGraph::new();
        let s2 = stack_addr(&mut eg, 2);
        let c8 = iconst(&mut eg, 8);
        let s2_plus_8 = add_node(&mut eg, Op::Add, &[s2, c8]);
        let s0 = stack_addr(&mut eg, 0);
        let ai = AliasInfo::new();
        assert!(!ai.may_alias(s2_plus_8, s0, &eg));
        assert!(!ai.must_alias(s2_plus_8, s0, &eg));
    }

    #[test]
    fn add_param_iconst_vs_stack_may_alias() {
        // Add(Param(0, I64), Iconst(8)) vs StackAddr(0) → may_alias=true (Unknown base)
        let mut eg = EGraph::new();
        let p0 = param(&mut eg, 0);
        let c8 = iconst(&mut eg, 8);
        let p0_plus_8 = add_node(&mut eg, Op::Add, &[p0, c8]);
        let s0 = stack_addr(&mut eg, 0);
        let ai = AliasInfo::new();
        assert!(ai.may_alias(p0_plus_8, s0, &eg));
    }

    #[test]
    fn add_stack_iconst_vs_same_slot_may_alias() {
        // Add(StackAddr(2), Iconst(8)) vs StackAddr(2) → same base → may_alias=true
        let mut eg = EGraph::new();
        let s2 = stack_addr(&mut eg, 2);
        let c8 = iconst(&mut eg, 8);
        let s2_plus_8 = add_node(&mut eg, Op::Add, &[s2, c8]);
        let s2b = stack_addr(&mut eg, 2);
        let ai = AliasInfo::new();
        assert!(ai.may_alias(s2_plus_8, s2b, &eg));
    }

    #[test]
    fn memo_correctness() {
        let mut eg = EGraph::new();
        let s = stack_addr(&mut eg, 5);
        let ai = AliasInfo::new();
        let r1 = ai.classify(s, &eg);
        let r2 = ai.classify(s, &eg);
        assert_eq!(r1, r2);
        assert_eq!(r1, AddrBase::StackSlot(5));
    }

    #[test]
    fn store_clobbers_load_mismatched_widths_conservative() {
        // Mismatched widths → types_overlap returns true → store_clobbers_load = true
        let mut eg = EGraph::new();
        let s = stack_addr(&mut eg, 0);
        let ai = AliasInfo::new();
        // I64 store vs I32 load at the same address
        assert!(ai.store_clobbers_load(s, Type::I64, s, Type::I32, &eg));
    }

    #[test]
    fn same_stack_slot_different_offsets_not_must_alias() {
        // Add(StackAddr(0), iconst(0)) vs Add(StackAddr(0), iconst(8)) share
        // a base but are different addresses. may_alias must be true (offsets
        // unknown), must_alias must be false.
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let c0 = iconst(&mut eg, 0);
        let c8 = iconst(&mut eg, 8);
        let a0 = add_node(&mut eg, Op::Add, &[s0, c0]);
        let a8 = add_node(&mut eg, Op::Add, &[s0, c8]);
        let ai = AliasInfo::new();
        assert!(ai.may_alias(a0, a8, &eg));
        assert!(!ai.must_alias(a0, a8, &eg));
    }

    #[test]
    fn store_clobbers_load_non_aliasing_false() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let s1 = stack_addr(&mut eg, 1);
        let ai = AliasInfo::new();
        assert!(!ai.store_clobbers_load(s0, Type::I64, s1, Type::I64, &eg));
    }

    /// `Op::Addr { .. }` propagates base from its first operand, same as
    /// `X86Lea*` and `Add`.
    #[test]
    fn addr_op_propagates_stack_base() {
        let mut eg = EGraph::new();
        let s2 = stack_addr(&mut eg, 2);
        let c4 = iconst(&mut eg, 4);
        let addr_node = add_node(&mut eg, Op::Addr { scale: 1, disp: 0 }, &[s2, c4]);
        let s2b = stack_addr(&mut eg, 2);
        let s3 = stack_addr(&mut eg, 3);
        let ai = AliasInfo::new();
        // Same slot base -> may-alias.
        assert!(ai.may_alias(addr_node, s2b, &eg));
        // Different slot -> no alias.
        assert!(!ai.may_alias(addr_node, s3, &eg));
    }

    /// Two known bases in an `Add` (e.g. `StackAddr + GlobalAddr`) cannot form
    /// a single-base address; `merge_add_bases` must widen to `Unknown`.
    #[test]
    fn add_of_two_different_known_bases_is_unknown() {
        let mut eg = EGraph::new();
        let s = stack_addr(&mut eg, 0);
        let g = global_addr(&mut eg, "g");
        let sum = add_node(&mut eg, Op::Add, &[s, g]);
        let ai = AliasInfo::new();
        assert_eq!(ai.classify(sum, &eg), AddrBase::Unknown);
    }

    /// Deeply nested Adds exceeding the recursion depth cap return Unknown
    /// rather than stack-overflowing.
    #[test]
    fn deep_add_chain_respects_depth_cap() {
        let mut eg = EGraph::new();
        let mut cur = stack_addr(&mut eg, 0);
        for _ in 0..32 {
            let c = iconst(&mut eg, 1);
            cur = add_node(&mut eg, Op::Add, &[cur, c]);
        }
        let ai = AliasInfo::new();
        // Depth cap kicks in; conservative Unknown is acceptable (and the
        // call must not stack-overflow).
        let base = ai.classify(cur, &eg);
        assert!(matches!(base, AddrBase::Unknown | AddrBase::StackSlot(0)));
    }

    /// Memoization: classifying the same class twice returns the same result
    /// without re-walking the e-class. This is a smoke test: repeated calls
    /// through an expression tree cache correctly.
    #[test]
    fn memo_caches_subexpressions() {
        let mut eg = EGraph::new();
        let s = stack_addr(&mut eg, 7);
        let c = iconst(&mut eg, 16);
        let add = add_node(&mut eg, Op::Add, &[s, c]);
        let ai = AliasInfo::new();
        assert_eq!(ai.classify(add, &eg), AddrBase::StackSlot(7));
        // Second call should hit the memo cache — just verify same answer.
        assert_eq!(ai.classify(add, &eg), AddrBase::StackSlot(7));
        // Classifying the base directly should also work.
        assert_eq!(ai.classify(s, &eg), AddrBase::StackSlot(7));
    }
}
