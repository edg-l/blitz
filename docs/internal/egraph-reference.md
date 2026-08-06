# E-Graph and Equality Saturation Reference

Core reference for Blitz's e-graph optimization infrastructure. Based on:
- "Fast and Extensible Equality Saturation" (Willsey et al., 2021) -- the egg paper
- "E-Graphs as a Persistent Compiler Abstraction" (Merckx et al., 2025, arXiv:2602.16707)
- "Cranelift's Acyclic E-Graph" (Fallin, 2026) -- aegraph blog post

## The Phase-Ordering Problem

Traditional compilers apply optimizations in a fixed sequence. Each phase destructively
rewrites the IR, committing to choices before later phases run. This means:

- Phase A might create a form that phase B could simplify, but B already ran
- Phase B might undo work that phase A did
- The optimal ordering depends on the input, so no fixed order is universally best

Equality saturation solves this by applying ALL rules simultaneously and deferring
the choice to cost-based extraction.

## Core Data Structures

### E-Node
An operation applied to zero or more child e-classes:
```
ENode { op: Op, children: SmallVec<[ClassId; 2]> }
```
Children MUST be canonicalized before hashing/insertion.

### E-Class
An equivalence class containing multiple e-nodes representing equivalent expressions:
```
EClass { id: ClassId, nodes: Vec<ENode>, ty: Type, analysis: AnalysisData }
```
All nodes in a class produce semantically equivalent values.

### Union-Find
Tracks equivalence relationships between e-classes. Supports:
- `find(id)` -- canonicalize with path compression, O(alpha(n))
- `union(a, b)` -- merge two classes, union by rank
- `find_immutable(id)` -- canonicalize without mutation

### Memo Table (Hash-Cons)
Maps canonicalized e-nodes to their e-class. Prevents duplicate nodes.
Enables O(1) lookup: "does this expression already exist?"

## The Equality Saturation Algorithm

### Single Unified Loop (correct approach)

```
loop {
    changed = false
    changed |= apply_algebraic_rules(egraph)
    changed |= apply_strength_reduction(egraph)
    changed |= apply_isel_rules(egraph)
    changed |= apply_addr_mode_rules(egraph)
    // ... all rule categories
    egraph.rebuild()    // enforce congruence closure
    if !changed { break }
}
```

ALL rules fire in EVERY iteration. This is the fundamental insight of equality
saturation: rules only ADD alternatives, never remove them. Running isel "too early"
costs nothing -- both the original and lowered forms exist, extraction picks cheapest.

### Why NOT Phased

Running rules in separate sequential phases (algebraic -> strength -> isel) reintroduces
the phase-ordering problem inside the e-graph. You lose:
- Isel creating forms that algebraic rules could simplify
- Strength reduction seeing results from algebraic simplification in the same iteration
- Cross-category interactions that only emerge when all rules fire together

### Rebuild (Congruence Closure)

After applying rules:
1. Drain memo table
2. Re-canonicalize all e-node children via union-find
3. If two e-nodes with same (op, canonical children) exist in different classes: merge
4. Repeat until worklist is empty

This propagates: if a = a' and b = b', then f(a,b) = f(a',b').

## E-Class Analyses

Per the egg paper, e-class analyses compute properties over equivalence classes that
guide rewrites and extraction.

### Requirements (Semilattice)

An analysis must provide:
- `make(enode) -> D` -- compute analysis data for a single e-node
- `merge(a: D, b: D) -> D` -- join when classes merge (must be commutative, associative, idempotent)
- Optional: `modify(class)` -- react to analysis changes (e.g., constant propagation)

### Key Analyses

**Constant Analysis**
- Domain: `Option<i64>` (None = not a constant, Some(v) = known constant)
- make: `Iconst(v, _) -> Some(v)`, everything else -> None
- merge: `merge(Some(a), Some(b)) = Some(a) if a==b, else None`; `merge(Some(a), None) = Some(a)`
- Replaces: manual `find_iconst()` scanning

**Known-Bits Analysis**
- Domain: `(known_zeros: u64, known_ones: u64)` -- bitmask of provably known bits
- make: `Iconst(v) -> (zeros=!v, ones=v)`, `And(a, mask) -> propagate`, `Shl(a, n) -> shift masks`
- merge: intersect known bits (both classes must agree)
- Enables: `And(x, 0xFF)` elimination after `Zext(i8)`, dead mask removal

**Type-Width Analysis**
- Domain: `(min_bits: u8, signed: bool)` -- minimum bits needed to represent value
- Enables: narrowing operations, avoiding unnecessary sign/zero extensions

## Cost Model and Extraction

### Cost Function
Each e-node has an intrinsic cost (latency, throughput, code size).
Generic IR ops (Add, Sub, Mul) have INFINITY cost -- they must be lowered.
x86-64 ops have finite costs from Agner Fog's instruction tables.

### Bottom-Up Extraction
1. Topological sort reachable classes (post-order DFS)
2. For each class bottom-up: pick e-node with min(own_cost + sum(child_costs))
3. Ties: prefer non-BlockParam

Extraction is the ONLY place where choices are made. Rules just add alternatives.

## Rewrite Rule Categories

### Algebraic Simplification
- Identity: `Add(a, 0) = a`, `Mul(a, 1) = a`, `Or(a, 0) = a`, `And(a, -1) = a`
- Annihilation: `Mul(a, 0) = 0`, `And(a, 0) = 0`
- Idempotence: `And(a, a) = a`, `Or(a, a) = a`
- Inverse: `Sub(a, a) = 0`, `Xor(a, a) = 0`
- Double negation: `Sub(0, Sub(0, a)) = a`
- Constant folding: `op(const1, const2) = const_result`
- Commutativity: `Add(a, b) = Add(b, a)` (canonical ordering to avoid blowup)
- Reassociation: `(a + b) + c = a + (b + c)` (enables constant folding through chains)

### Strength Reduction
- `Mul(a, 2^n) = Shl(a, n)`
- `Mul(a, 3/5/9) = Add(a, Shl(a, 1/2/3))`
- `UDiv(a, 2^n) = Shr(a, n)`
- `URem(a, 2^n) = And(a, 2^n - 1)`
- `SDiv(a, 2^n)` -> arithmetic shift pattern

### Instruction Selection (x86-64)
- ALU: `Add -> X86Add`, `Sub -> X86Sub`, etc.
- Shifts: `Shl -> X86Shl`, with immediate variants
- Compares: `Icmp -> Proj1(X86Sub)` (flags reuse)
- Division: `SDiv/SRem -> Proj0/Proj1(X86Idiv)` (shared instruction)
- Floating point: `Fadd -> X86Addsd/X86Addss`, etc.
- Conversions: `Sext -> X86Movsx`, `Zext -> X86Movzx`, `Trunc -> X86Trunc`
- Select: `Select(flags, t, f) -> X86Cmov(cc, flags, t, f)`

### Addressing Modes
- `Add(base, disp) -> Addr{1, disp}(base, NONE)`
- `Add(base, Shl(idx, n)) -> Addr{2^n, 0}(base, idx)`
- LEA formation: `X86Lea2`, `X86Lea3{scale}`, `X86Lea4{scale, disp}`

## Constrained Extraction

### `extract_at` and `extract_at_with_memo`

```rust
pub fn extract_at(
    egraph: &EGraph,
    class: ClassId,
    live_classes: &BTreeSet<ClassId>,
    cost_model: &CostModel,
) -> Option<ExtractedNode>

pub fn extract_at_with_memo(
    egraph: &EGraph,
    class: ClassId,
    live_classes: &BTreeSet<ClassId>,
    cost_model: &CostModel,
    memo: &BTreeMap<ClassId, ExtractedNode>,
) -> Option<ExtractedNode>
```

Cost-aware re-extraction constrained to a set of classes that are already live
at a given program point. Used by the pressure-driven live-range splitter (Phase 5)
to determine whether a value can be rematerialized at a split point.

**Parameters:**
- `class`: the e-class to extract a best node for.
- `live_classes`: the set of canonical `ClassId`s whose values are live at the target
  program point. A child class in this set contributes 0 to the total cost; a child
  not in this set uses its pre-computed `memo` cost.
- `cost_model`: the same `CostModel` used by the main extraction pass.
- `memo`: (`extract_at_with_memo` only) the `ExtractionResult::choices` from a prior
  full [`extract`] call covering at least all classes reachable from `class`.

**Return value:**
Returns the `ExtractedNode` with the lowest total cost `own_cost + sum(child_cost)`,
or `None` if no candidate node has a finite total cost.

**Free remat ops** (`Iconst`, `Fconst`, `StackAddr`, `GlobalAddr`, `Param`,
`BlockParam`) have zero cost and no children. They are always selectable regardless
of `live_classes`.

**Cost semantics:**
- If a child is in `live_classes`: child cost = 0.
- Otherwise: child cost = `memo[child].cost` (the full bottom-up extraction cost).
- If a required child has no memo entry, that candidate node is skipped.

**Preferred entry point:** use `extract_at_with_memo` when the caller already has
the full `ExtractionResult`. `extract_at` is a thin wrapper that runs `extract`
internally first.

**Invariant:** when `live_classes` is a superset of all classes reachable from
`class`, `extract_at_with_memo` picks the same op as the standard `extract` pass
for that class.

**Source:** `src/egraph/extract.rs` — `extract_at`, `extract_at_with_memo`.

## Saturation Control

- **Iteration limit**: Cap total iterations to prevent infinite loops
- **Class count limit**: Abort if e-graph exceeds max_classes (default 500k)
- **Changed flag**: Exit early when no rules fire (fixpoint reached)
- **Scheduling**: Can prioritize rules that are likely to make progress (future work)
