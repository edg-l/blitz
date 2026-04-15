# LICM Implementation Plan

Loop-Invariant Code Motion for Blitz. Identifies pure computations inside loops whose transitive e-graph dependencies are all outside the loop, hoists them into a preheader block so they execute once instead of every iteration. Runs before e-graph optimization.

## Pipeline Placement

```
Inlining -> LICM (CFG transform on &mut Function + &mut EGraph) -> E-graph optimization -> Extraction -> Linearization -> Scheduling -> Regalloc -> ...
```

## Phase 1: Infrastructure and Plumbing [DONE]

- [x] Add `enable_licm: bool` to `CompileOptions` (default false)
- [x] Add `--enable-licm` CLI flag to tinyc, thread opts via `_with_opts` variants
- [x] Add `next_block_id: BlockId` to `Function`, populate in `FunctionBuilder::finalize()`, add `fresh_block_id()` helper
- [x] Restructure `compile()`: take egraph out first, call `run_licm(&mut func, &mut egraph)` when enabled, pass `extra_roots` into linearization loop
- [x] Same restructure for `compile_to_ir_string()`
- [x] Create `src/compile/licm.rs` with `ExtraRoots` type and stub `run_licm()`
- [x] Checkpoint: cargo check + cargo test + lit tests all pass

## Phase 2: Loop Detection

- [ ] `detect_back_edges(func, rpo, idom) -> Vec<(usize, usize)>` in `licm.rs`. Back-edge = edge where target dominates source (proper dominator-based check using `dominates()` from `cfg.rs`).
- [ ] `collect_loop_body(header_idx, back_edge_src, preds) -> BTreeSet<usize>`. Backward predecessor walk from back-edge source to header.
- [ ] `build_predecessor_map(func) -> (Vec<Vec<usize>>, BTreeMap<BlockId, usize>)`. Returns `(preds, id_to_idx)`.
- [ ] `LoopInfo { header_idx, body: BTreeSet<usize>, back_edges: Vec<(usize, usize)> }` struct.
- [ ] `detect_loops(func) -> Vec<LoopInfo>`. Calls compute_rpo, compute_idom, detect_back_edges, groups by header, union of bodies. Sorted by header RPO position (outermost first).
- [ ] Unit tests: simple while loop, nested loops, no loops, self-loop.
- [ ] Checkpoint: cargo test passes.

## Phase 3: Preheader Insertion

- [ ] `insert_preheader(func, egraph, loop_info, id_to_idx) -> usize`. Allocates fresh BlockId via `func.fresh_block_id()`. Creates BlockParam e-graph nodes via `egraph.add()` for preheader params (same param_types as header). Preheader Jump targets the header, forwarding its BlockParam ClassIds. Redirects non-back-edge predecessors to the preheader.
- [ ] `redirect_predecessors(func, old_target, new_target, pred_indices)`. Rewrites Jump/Branch terminators.
- [ ] Wire into `run_licm()`: for each loop (innermost-first), call `insert_preheader()`.
- [ ] Update `run_licm` signature: `pub fn run_licm(func: &mut Function, egraph: &mut EGraph) -> ExtraRoots` (already done).
- [ ] Unit tests: preheader inserted correctly, predecessors redirected, param_types match header.
- [ ] Checkpoint: cargo test passes.

## Phase 4: Invariant Detection

- [ ] `is_class_loop_invariant(class_id, egraph, loop_body, header_id, loop_defined_classes, cache) -> bool`. A class is invariant if: not a BlockParam of any loop body block, not effectful (LoadResult/CallResult/StoreBarrier/VoidCallBarrier), and all e-graph node children are recursively invariant.
- [ ] `collect_loop_defined_classes(func, egraph, loop_body) -> BTreeSet<ClassId>`. All ClassIds from effectful ops in loop body blocks (use `push_block_class_ids` pattern) + BlockParam ClassIds for loop body blocks. Canonicalize via `egraph.unionfind.find_immutable()`.
- [ ] `find_invariant_classes(func, egraph, loop_info) -> Vec<ClassId>`. Fixed-point: collect candidate ClassIds referenced by loop body, check each with `is_class_loop_invariant`.
- [ ] Unit tests: iconst is invariant, add(param, iconst) is invariant, add(block_param_of_header, iconst) is NOT, LoadResult is NOT.
- [ ] Checkpoint: cargo test passes.

## Phase 5: Hoisting and Integration

- [ ] Wire `find_invariant_classes()` into `run_licm()`. For each loop with a preheader, add invariant ClassIds to `ExtraRoots[preheader_idx]`.
- [ ] ExtraRoots integration already done in compile() and compile_to_ir_string() (Phase 1).
- [ ] LICM tracing: `BLITZ_DEBUG=licm` logs detected loops, invariant count, hoisted classes.
- [ ] Edge cases: no loops (early return), entry block as header (skip preheader insertion).
- [ ] Checkpoint: cargo test passes.

## Phase 6: Lit Tests

- [ ] `tests/lit/licm/simple_hoist.c` - pure computation hoisted before loop
- [ ] `tests/lit/licm/no_hoist_load.c` - effectful op stays in loop
- [ ] `tests/lit/licm/no_hoist_loop_dep.c` - loop-carried dep stays in loop
- [ ] `tests/lit/licm/nested_loop.c` - hoist to outermost preheader
- [ ] `tests/lit/licm/correctness_sum.c` - end-to-end OUTPUT test
- [ ] Full regression suite (all existing lit tests pass with LICM off)

## Phase 7: Polish

- [ ] Consider enabling by default (only if no regressions)
- [ ] Doc comments, clippy, fmt
- [ ] Final audit: every task implemented, no stubs

## Key Design Decisions

- **`&mut EGraph` passed to LICM** so we can `egraph.add()` BlockParam nodes for preheaders
- **Borrow checker solution**: take egraph out of func first, pass both separately
- **Hoisting = extra roots**: invariant ClassIds added to preheader's root list during linearization; RPO ordering ensures they emit before the loop body
- **Invariant check walks raw e-graph children** (not extraction); pre-optimization, each class typically has one node
- **Loop bodies stored as `BTreeSet<usize>`** (block indices); stable because preheaders are appended (no index shifts)
- **Effectful ops checked**: LoadResult, CallResult, StoreBarrier, VoidCallBarrier
- **Preheader BlockParam nodes**: get distinct ClassIds via `egraph.add()` with unique `(block_id, param_idx)` tuples, preventing e-graph merging with header params
- **Back-edge detection**: dominator-based (target dominates source), not the simpler index-comparison heuristic in `compute_loop_depths`

## Edge Cases

- **Entry block is loop header**: skip preheader insertion (no predecessors to redirect)
- **Multiple back-edges to same header**: group them, union of bodies, one preheader
- **Irreducible control flow**: tinyc produces reducible CFGs; irreducible loops have no natural loop and are safely skipped
- **E-graph merging after LICM**: hoisted class stays rooted in preheader; merging doesn't change that
- **Empty loops**: no invariants to hoist, harmless preheader insertion
- **Hoisting increases register pressure**: inherent LICM tradeoff; future work could add pressure heuristic

## Files

- `src/compile/licm.rs` - all LICM logic (loop detection, preheader insertion, invariant analysis)
- `src/compile/mod.rs` - pipeline integration, CompileOptions
- `src/compile/ir_print.rs` - IR output pipeline integration
- `src/ir/function.rs` - `next_block_id` field, `fresh_block_id()` helper
- `src/ir/builder.rs` - populate `next_block_id` in finalize()
- `crates/tinyc/src/main.rs` - `--enable-licm` CLI flag
- `crates/tinyc/src/lib.rs` - `_with_opts` function variants
- `tests/lit/licm/*.c` - lit tests
