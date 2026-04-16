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

## Phase 2: Loop Detection [DONE]

- [x] `detect_back_edges(func, rpo, idom) -> Vec<(usize, usize)>` in `licm.rs`. Back-edge = edge where target dominates source (proper dominator-based check using `dominates()` from `cfg.rs`).
- [x] `collect_loop_body(header_idx, back_edge_src, preds) -> BTreeSet<usize>`. Backward predecessor walk from back-edge source to header.
- [x] `build_predecessor_map(func) -> (Vec<Vec<usize>>, BTreeMap<BlockId, usize>)`. Returns `(preds, id_to_idx)`.
- [x] `LoopInfo { header_idx, body: BTreeSet<usize> }` struct.
- [x] `detect_loops(func) -> Vec<LoopInfo>`. Calls compute_rpo, compute_idom, detect_back_edges, groups by header, union of bodies. Sorted by header RPO position (outermost first).
- [x] Unit tests: simple while loop, nested loops, no loops, self-loop (11 tests).
- [x] Checkpoint: cargo test passes.

## Phase 3: Preheader Insertion [DONE]

- [x] `insert_preheader(func, egraph, loop_info, id_to_idx) -> usize`. Allocates fresh BlockId via `func.fresh_block_id()`. Creates BlockParam e-graph nodes via `egraph.add()` for preheader params (same param_types as header). Preheader Jump targets the header, forwarding its BlockParam ClassIds. Redirects non-back-edge predecessors to the preheader.
- [x] `redirect_predecessors(func, old_target, new_target, pred_indices)`. Rewrites Jump/Branch terminators.
- [x] Entry block as header edge case: returns header_idx without mutation.
- [x] Unit tests: preheader inserted correctly, predecessors redirected, param_types match header, entry-block-as-header (4 tests).
- [x] Checkpoint: cargo test passes.

## Phase 4: Invariant Detection [DONE]

- [x] `collect_loop_defined_classes(func, egraph, loop_body) -> BTreeSet<ClassId>`. Collects Load results, Call results, and BlockParam ClassIds for loop body blocks. Only outputs (not operands/uses).
- [x] `is_class_loop_invariant(class_id, egraph, loop_defined, cache) -> bool`. A class is invariant if: not in loop_defined, and at least one node is non-effectful with all children recursively invariant. Memoized via HashMap cache.
- [x] `find_invariant_classes(func, egraph, loop_info) -> Vec<ClassId>`. Transitive e-graph walk from effectful op operands; collects all invariant classes reachable from the loop body.
- [x] `collect_effectful_operands(func, block_indices) -> Vec<ClassId>`. Helper for scanning effectful op operands.
- [x] Unit tests: iconst invariant, add(param, iconst) invariant, block_param NOT invariant, LoadResult NOT invariant, add with loop-defined child NOT invariant (5 tests).
- [x] Checkpoint: cargo test passes.

## Phase 5: Hoisting and Integration [DONE]

- [x] Wire `find_invariant_classes()` into `run_licm()`. For each loop with a preheader, add invariant ClassIds to `ExtraRoots[preheader_idx]`.
- [x] ExtraRoots integration already done in compile() and compile_to_ir_string() (Phase 1).
- [x] LICM tracing: `BLITZ_DEBUG=licm` logs detected loops, per-loop stats, total hoisted count.
- [x] Edge cases: no loops (early return), entry block as header (skip preheader insertion).
- [x] Checkpoint: cargo test passes.

## Phase 6: Lit Tests [DONE]

- [x] `tests/lit/licm/simple_hoist.c` - pure computation hoisted before loop, correctness (EXIT: 70)
- [x] `tests/lit/licm/no_hoist_load.c` - effectful ops stay in loop, correctness (EXIT: 55)
- [x] `tests/lit/licm/no_hoist_loop_dep.c` - loop-carried dep stays in loop, correctness (EXIT: 55)
- [x] `tests/lit/licm/nested_loop.c` - nested loops handled correctly (EXIT: 30)
- [x] `tests/lit/licm/correctness_sum.c` - end-to-end with inlined function (OUTPUT: pass)
- [x] Full regression suite: 333 lit tests pass, 591 unit + 61 codegen pass

## Phase 7: Polish [DONE]

- [x] Clippy clean (collapsed nested if-let, removed unused `back_edges` field)
- [x] cargo fmt applied
- [x] Final audit: every task implemented, no stubs
- [x] Fixed inliner bug: `next_block_id` not updated after `inline_call_site` (caused BlockId collision with LICM)
- [ ] Consider enabling by default (deferred; opt-in via `--enable-licm` for now)

## Key Design Decisions

- **`&mut EGraph` passed to LICM** so we can `egraph.add()` BlockParam nodes for preheaders
- **Borrow checker solution**: take egraph out of func first, pass both separately
- **Hoisting = extra roots**: invariant ClassIds added to preheader's root list during linearization; RPO ordering ensures they emit before the loop body
- **Invariant check walks raw e-graph children transitively** (not extraction); pre-optimization, each class typically has one node
- **Loop bodies stored as `BTreeSet<usize>`** (block indices); stable because preheaders are appended (no index shifts)
- **Effectful ops checked**: LoadResult, CallResult, StoreBarrier, VoidCallBarrier
- **Preheader BlockParam nodes**: get distinct ClassIds via `egraph.add()` with unique `(block_id, param_idx)` tuples, preventing e-graph merging with header params
- **Back-edge detection**: dominator-based (target dominates source), not the simpler index-comparison heuristic in `compute_loop_depths`
- **loop_defined = only outputs**: Load results, Call results, BlockParams. Operands (addr, val, cond, args) are uses, not definitions, and may be loop-invariant.
- **Transitive candidate walk**: `find_invariant_classes` walks e-graph children transitively from effectful op operands to find invariant subexpressions (e.g., `a+b` where `a` and `b` are loop-invariant params).

## Bugs Found and Fixed

- **Inliner `next_block_id` bug**: `inline_call_site` in `src/inline/transform.rs` did not update `caller.next_block_id` after appending remapped + continuation blocks. Fixed by adding `caller.next_block_id = caller.next_block_id.max(cont_id + 1)` after block insertion. This caused BlockId collisions when LICM's `fresh_block_id()` was called after inlining.

## Edge Cases

- **Entry block is loop header**: skip preheader insertion (no predecessors to redirect)
- **Multiple back-edges to same header**: group them, union of bodies, one preheader
- **Irreducible control flow**: tinyc produces reducible CFGs; irreducible loops have no natural loop and are safely skipped
- **E-graph merging after LICM**: hoisted class stays rooted in preheader; merging doesn't change that
- **Empty loops**: no invariants to hoist, harmless preheader insertion
- **Hoisting increases register pressure**: inherent LICM tradeoff; future work could add pressure heuristic

## Files

- `src/compile/licm.rs` - all LICM logic (loop detection, preheader insertion, invariant analysis, hoisting)
- `src/compile/mod.rs` - pipeline integration, CompileOptions
- `src/compile/ir_print.rs` - IR output pipeline integration
- `src/ir/function.rs` - `next_block_id` field, `fresh_block_id()` helper
- `src/ir/builder.rs` - populate `next_block_id` in finalize()
- `src/inline/transform.rs` - fix `next_block_id` after inlining
- `src/trace.rs` - added `licm` debug category
- `crates/tinyc/src/main.rs` - `--enable-licm` CLI flag
- `crates/tinyc/src/lib.rs` - `_with_opts` function variants
- `tests/lit/licm/*.c` - 5 lit tests
