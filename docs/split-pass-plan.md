# Pressure-Driven Live-Range Splitting Pass (Plan)

## Overview

Replace the global allocator's iterative spill loop and its post-hoc rename plumbing (`coalesce_aliases`, `per_block_rename_maps`, `vreg_slot`, `vreg_remat_op`) with a pre-allocation IR pass that splits infeasible live ranges into multiple VRegs, producing an input the Chaitin-Briggs allocator can color in a single pass. The splitter uses the retained e-graph to perform cost-aware re-extraction of a value at each split point, which is uniquely clean in this architecture and the core justification for the refactor.

## Requirements

- Explicit
  - Fix `tests/lit/regalloc/xmm_loop_crossing.c` (expected `11.250000`).
  - Preserve all behavioral coverage of the 627 unit tests — functional equivalents for any deleted function tests MUST be added in Phase 5 or 6 BEFORE the deletion in 7B. Expected end count: 653 unit tests (627 + 1 Phase 1 + 3 Phase 2 + 5 Phase 3 + 7 Phase 4 + 6 Phase 5 + 4 Phase 6 + 15 ported in 7.4-pre − 15 deleted in 7B).
  - Preserve all 357 currently-passing lit tests.
  - Each phase leaves the tree green; no broken commits.
  - Complete removal of `coalesce_aliases`, `per_block_rename_maps`, `vreg_slot`, `vreg_remat_op`, `run_phase5`'s spill-and-recolor loop, `insert_spills_global`, `select_spill_candidates_global`, `select_spill_by_phantom_interference`, `rebuild_interference`, `compute_global_liveness_with_block_params`'s alias-renaming dance, and `src/regalloc/split.rs` by end of rollout.
  - New module at `src/compile/split.rs` (pressure splitter). Distinct from the dead old file `src/regalloc/split.rs`.
  - New e-graph helper `egraph::extract::extract_at(class, point, live_set)` reuses the existing `CostModel`.
- Inferred
  - The splitter must decide per live range whether to (a) remat at use (preferred when a cheap remat op exists, e.g. `Fconst`, `Iconst`, `StackAddr`, `GlobalAddr`), (b) insert a cross-call slot-backed spill+reload, or (c) leave alone. This mirrors the choice currently made inside `insert_spills_global`, but decides it up-front with e-graph help instead of after-the-fact.
  - Program points must be stable and total-ordered across the function. We use `(block_idx, inst_idx)` within the function-scope schedule (flattened indices also computed). Barrier operands count at their barrier's position.
  - The split pass runs AFTER scheduling and effectful-op operand population (so program points correspond to final scheduled positions) but BEFORE `allocate_global`.
- Assumptions
  - The existing `CostModel` cost function is sufficient for split-time re-extraction (no new tuning table required). If per-point live-set constraints change extraction choices, they do so via feasibility pruning only; the cost metric itself is unchanged.
  - The global allocator's spill loop is the only place where spills originate after this refactor. Phase 5 `insert_early_barrier_spills` (compile/mod.rs:809) stays as-is; it is a pre-pass that only stores high-churn barrier results and is unrelated to pressure splitting.
  - A VReg defined by `Op::BlockParam` cannot be split by copying the def site (it has no explicit def instruction). Handling is described in Phase 6.

## Architecture Decision

Recommended approach: Implement a range-keyed `class_to_vreg` map and a single splitting pass that operates on the already-scheduled per-block `Vec<ScheduledInst>`. The pass scans each block forward, tracks live pressure per register class, picks victims when pressure exceeds budget, and either remats or inserts a `SpillStore`/`SpillLoad` pair that defines a fresh VReg. The e-graph's `extract_at` is consulted only when the pass wants an alternative materialization (remat) that the original class may no longer offer at this point (e.g. the original op was folded away by saturation).

Alternatives considered and rejected:

- SSA-form live-range splitting a la LLVM greedy regalloc: implements interval trees, Hopcroft-Karp-style hopping, and a large interval DB. Overkill for Blitz's size and complicates the per-class pressure model already used here. Rejected.
- Keep the spill loop but add a pre-pass that only inserts remats: does not solve the block-param-rename fights. Rejected.
- Put the split pass inside `allocate_global`: obscures the contract (allocator should not mutate IR names) and preserves the rename plumbing. Rejected.

Tradeoff accepted: the split pass must re-run liveness internally to drive its pressure metric, duplicating some work with `compute_global_liveness`. Acceptable because the splitter's per-block backward scan is cheap and reuses `GlobalLiveness::live_in` as its seed so pass-through liveness is captured correctly.

## Consumers of `class_to_vreg` that must take a program point after refactor

Inventoried via `rg class_to_vreg src/`. Each listed consumer is a `BTreeMap<ClassId, VReg>` lookup that must go through the new `lookup(class, program_point)` API once the map becomes range-keyed. Entries within a single function are grouped; parameter name is the argument on the callee that holds the map.

- `src/egraph/extract.rs:254,267,273,281,292,297,312,319,327,351,364,374,382,399,403` — construction sites. Change emit logic to build range-keyed entries and expose `ClassToVRegMap::lookup(class, point)` + `insert(class, (vreg, start, end))`.
- `src/ir/print.rs:171,184,189,190,194,195,206,210,227,230,234,245,250,261,310,318` — IR printer. Printer receives a program-point-less map for readability; pre-flatten to "first VReg per class" or show ranges inline. Task: add `for_printing()` accessor on the new map.
- `src/compile/mod.rs:331,383,405,407,417,477,486,495,536,554,638,647,673,679,696,701,715,759,769,786,805,836,843,855,1005,1013,1044,1061,1190,1224,1256,1257` — pipeline orchestration. Every site must take a program point:
  - `:331` construction site (Phase 3 of current pipeline, at VReg linearization)
  - `:383,405,477,486,490` snapshot/restore cross-block view
  - `:407` calls `vreg_insts_for_block` (must return range-keyed entries)
  - `:417` post-linearization fixup (block param) — already per-block, carries `block_idx` as the point
  - `:495` `build_vreg_types` consumer — iterates keys, now iterates `(class, vreg, range)` entries
  - `:536,554,673,679,805,836,843` barrier context building — passes the barrier's program point
  - `:638,647,696,701,715,759,769,786,855` param/live-out collection — passes "block entry" as point
  - `:1005-1061` block-local lowering. The whole `block_class_to_vreg` construction (snapshot + renames + overrides + aliases) is DELETED in Phase 7. Replaced with direct `lookup(class, (block_idx, inst_idx))` at each call site.
  - `:1190,1224,1256,1257` lowering to `MachInst`
- `src/compile/cfg.rs:306,320,330,345,354,358` — `collect_phi_source_vregs`, `compute_copy_pairs`. Consumers pass `"end of block"` as the point for terminator VRegs (the copy happens at block exit) and `"start of target block"` for the param side.
- `src/compile/terminator.rs:174,175,178,193,231,232,235,267,268,271,279,280,283,347,348,351,386,391,407,410,416,418` — phi-copy emission. Caller passes `block_end` for args and `target_block_start` for params. `coalesce_aliases` parameter removed entirely in Phase 7 along with the alias-chase loop at `:418-423`.
- `src/compile/barrier.rs:19,24,41,45,50,60,67,76,88,351,380,402,420` — `build_barrier_maps`, `populate_effectful_operands`. Each non-terminator effectful op has a barrier position; map is queried at that position.
- `src/compile/precolor.rs:18,33,113,135,155` — `add_call_precolors_for_block`, shift/div precolors. All use the barrier's point (Call) or instruction's point (Shift/Div).
- `src/compile/effectful.rs` (19 call sites at `:30,39,54,60,95,104,123,139,198,217,257,357,365,375,390,428,471,490,495,496`) — Load/Store/Call argument resolution. Uses the effectful op's barrier position. This is the highest-density consumer file outside `mod.rs`.
- `src/compile/ir_print.rs:44,67,72,123,152,197` — IR printer (compile-time dump). Same treatment as `ir/print.rs`.
- `src/regalloc/global_liveness.rs:186,213,291,300,345,364` — `compute_phi_uses`, `collect_block_param_vregs_per_block`, `apply_block_param_overrides_to_phi_uses`. Uses block-edge points; `apply_block_param_overrides_to_phi_uses` is deleted entirely in Phase 7 as overrides are subsumed by splits.

Non-consumer: `src/regalloc/mod.rs:63` is a doc-comment reference, no code change.

## Implementation Plan

### Phase 1: Range-keyed `class_to_vreg` (read-compat wrapper, no behavior change)

- Goal: introduce a `ClassVRegMap` type that wraps the current `BTreeMap<ClassId, VReg>` behind a narrow API, without changing callers. Sets up later phases for per-point lookups.
- Concrete changes:
  - `src/egraph/extract.rs`: introduce `pub struct ClassVRegMap { single: BTreeMap<ClassId, VReg> }` with methods `new()`, `insert_single(class, vreg)`, `lookup_single(class) -> Option<VReg>`, `iter() -> impl Iterator<Item = (ClassId, VReg)>`, `keys()`, `contains(class)`. Do NOT add `lookup(class, point)` yet; the point parameter is introduced in Phase 3.
  - Change `extraction_to_vreg_insts_with_map` and `vreg_insts_for_block` to return/accept `&mut ClassVRegMap`.
  - All 19 files using `class_to_vreg: &BTreeMap<ClassId, VReg>` change their parameter type to `&ClassVRegMap`. Call sites swap `.get(&canon)` for `.lookup_single(canon)`.
  - `build_vreg_types` uses `.iter()`.
  - Add a single `Deref` shim ONLY if needed for `ir/print.rs` and `compile/ir_print.rs`; prefer explicit `.iter()` usage. No public `BTreeMap` leakage.
- Test milestone:
  - All 627 unit tests pass.
  - All 357 lit tests pass (same baseline).
  - Add 1 unit test in `src/egraph/extract.rs`: `classvregmap_single_insert_lookup` verifying the new API round-trips.
- Risk callout: mechanical rename touches many files (~20). Likely failure: signature mismatch after find-replace. Mitigation: use `cargo check` as the driver, fix call by call. Keep the wrapper a struct (not tuple struct) so method resolution is unambiguous. Do not change any behavior; this is purely a naming indirection.
- Dependencies: none.

- [ ] Task 1.1: Add `ClassVRegMap` struct + methods in `src/egraph/extract.rs` (Complexity: Low)
- [ ] Task 1.2: Update `extraction_to_vreg_insts_with_map`, `extraction_to_vreg_insts`, `vreg_insts_for_block`, `build_vreg_types` signatures to use `ClassVRegMap` (Complexity: Low)
- [ ] Task 1.3: Propagate type through `src/compile/mod.rs` (32 sites) (Complexity: Medium)
- [ ] Task 1.4: Propagate through the following per-file consumers, each named explicitly: `src/compile/barrier.rs`, `src/compile/cfg.rs`, `src/compile/effectful.rs` (19 call sites — highest density), `src/compile/ir_print.rs`, `src/compile/precolor.rs`, `src/compile/terminator.rs` (Complexity: Medium)
- [ ] Task 1.5: Propagate through `src/regalloc/global_liveness.rs` + `src/ir/print.rs` (Complexity: Low)
- [ ] Task 1.6: Add unit test `classvregmap_single_insert_lookup` in `src/egraph/extract.rs` (Complexity: Low)
- [ ] Task 1.7: Run `cargo fmt`, `cargo clippy`, `cargo test`, `bash tests/lit/run_tests.sh` — all green (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 1 complete** — Review tasks 1.1-1.7. Confirm every `BTreeMap<ClassId, VReg>` parameter on non-private APIs now reads `ClassVRegMap`. Confirm both test suites pass at the prior baseline (627 unit + 1 new = 628). Do not proceed until green.

### Phase 2: Program-point type and flat indexing

- Goal: introduce `ProgramPoint` (block_idx, inst_idx) with a total order and helpers to map from barriers/terminators to points. No consumer switches yet.
- Concrete changes:
  - New file `src/compile/program_point.rs`:
    - `#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)] pub struct ProgramPoint { pub block: u32, pub inst: u32 }`
    - `ProgramPoint::BLOCK_ENTRY(block)`, `ProgramPoint::BLOCK_EXIT(block)` constants (use `inst = 0` and `inst = u32::MAX`).
    - `pub fn barrier_point(block_idx: usize, barrier_idx: usize, schedule: &[ScheduledInst]) -> ProgramPoint` — maps a barrier op's position in `block.ops` to its sched index.
  - Add `pub mod program_point;` to `src/compile/mod.rs`.
  - Add 3 unit tests in the new file: `ordering`, `entry_exit_distinct`, `barrier_point_finds_barrier_result`.
- Test milestone:
  - 628 + 3 new unit tests pass = 631.
  - 357 lit tests pass.
- Risk callout: inst index definition must match exactly how the splitter and consumers use it. Using `inst: u32::MAX` as exit is fine because schedules are under 10k instructions in practice; add a `debug_assert!(inst < u32::MAX - 1)` in `ScheduledInst` construction is NOT added (too disruptive); instead a debug assert inside `barrier_point`.
- Dependencies: Phase 1.

- [ ] Task 2.1: Create `src/compile/program_point.rs` with struct + constants + `barrier_point` (Complexity: Low)
- [ ] Task 2.2: Add 3 unit tests for `ProgramPoint` ordering and barrier mapping (Complexity: Low)
- [ ] Task 2.3: Register module in `src/compile/mod.rs` (Complexity: Low)
- [ ] Task 2.4: Run `cargo fmt`, `cargo test`, lit suite — all green (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 2 complete** — `ProgramPoint` type compiles, tests pass, nothing else changed.

### Phase 3: `extract_at(class, point, live_set)` on the e-graph

- Goal: add a cost-aware re-extraction helper that, given a class and a program point's live operand set, returns the best `ExtractedNode` whose transitive children's roots are all either (a) already live at `point` or (b) cheaply reconstructible constants/free ops (`Iconst`, `Fconst`, `StackAddr`, `GlobalAddr`, `Param`, `BlockParam`).
- Concrete changes:
  - In `src/egraph/extract.rs`, add:
    ```
    pub fn extract_at(
        egraph: &EGraph,
        class: ClassId,
        live_classes: &BTreeSet<ClassId>,
        cost_model: &CostModel,
    ) -> Option<ExtractedNode>
    ```
  - Implementation: iterate nodes in `egraph.class(class).nodes`; for each candidate op, compute `own_cost + sum(child_remat_cost)` where `child_remat_cost` is 0 if `child in live_classes`, else the precomputed bottom-up extract cost for that child from a standard `extract` pass memo reused here. If no node yields a finite total cost, return `None`.
  - Reuses `CostModel` unchanged. Accepts existing `memo` by optionally taking `&BTreeMap<ClassId, ExtractedNode>` (the full extraction result); if caller has it, skip recompute. Signature:
    ```
    pub fn extract_at_with_memo(
        egraph: &EGraph,
        class: ClassId,
        live_classes: &BTreeSet<ClassId>,
        cost_model: &CostModel,
        memo: &BTreeMap<ClassId, ExtractedNode>,
    ) -> Option<ExtractedNode>
    ```
    and `extract_at` is a thin wrapper that calls `extract` internally first. In the splitter we'll use the memoized form (the pipeline already has an `ExtractionResult`).
  - Special-case free remat ops: nodes whose own cost is 0 AND have no children (`Iconst`, `Fconst`, `StackAddr`, `GlobalAddr`, `Param`, `BlockParam`) are always selectable; children live-set is irrelevant.
- Test milestone:
  - 5 new unit tests in `extract.rs`:
    - `extract_at_prefers_free_remat` (Iconst selectable when not in live set)
    - `extract_at_uses_live_child` (an X86Add where both children live)
    - `extract_at_rejects_nonlive_expensive_children` (a deep tree with no live internals returns None)
    - `extract_at_matches_extract_when_all_live` (agreement with normal extract when everything is live)
    - `extract_at_returns_none_on_truly_infeasible` (class with only generic-IR nodes returns None)
  - 631 + 5 unit tests pass (= 636), lit suite holds.
- Risk callout: cost semantics drift. If `extract_at` picks a different op than `extract` in the full-live case, downstream extraction invariants break. Mitigation: unit test `extract_at_matches_extract_when_all_live` runs on Programs 1-5 of the existing `extraction_correctness_five_programs`. If any program diverges, we rewrite `extract_at` to delegate to `extract` when `live_classes` is a superset of reachable classes.
- Dependencies: Phase 2 (for `ProgramPoint`, though not used directly here; splitter consumes both).

- [ ] Task 3.1: Add `extract_at` and `extract_at_with_memo` in `src/egraph/extract.rs` (Complexity: Medium)
- [ ] Task 3.2: Add 5 unit tests listed above (Complexity: Medium)
- [ ] Task 3.3: Document `extract_at` API reference (function signatures, parameter semantics, return invariants) in `docs/egraph-reference.md` under a new "Constrained extraction" section. Cross-linking to `docs/split-pass.md` is handled in Phase 8.3. (Complexity: Low)
- [ ] Task 3.4: Run `cargo fmt`, `cargo test`, lit suite — green (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 3 complete** — `extract_at` exists, 5 new tests pass (baseline 636 unit), docs updated. Agreement property holds on 5-program correctness set.

### Phase 4: Multi-VReg range-keyed class map (shadow-only, no splits produced yet)

- Goal: upgrade `ClassVRegMap` to `BTreeMap<ClassId, Vec<(VReg, ProgramPoint, ProgramPoint)>>` form internally. Consumers switch to `lookup(class, point)`. Today every class has a single entry spanning `BLOCK_ENTRY(0)..=BLOCK_EXIT(last)`, so behavior is unchanged.
- Concrete changes:
  - `src/egraph/extract.rs`:
    - Replace `ClassVRegMap::single` with `segments: BTreeMap<ClassId, SmallVec<[Segment; 2]>>` where `struct Segment { vreg: VReg, start: ProgramPoint, end: ProgramPoint }`.
    - Methods: `insert_full_range(class, vreg)`, `insert_segment(class, vreg, start, end)`, `truncate_segment_start(vreg, new_start)` (shrinks an existing segment on the low side; updates both forward `segments` storage and the inverse `vreg_to_class` index atomically, so stale entries are impossible), `lookup(class, point) -> Option<VReg>`, `lookup_any(class) -> Option<VReg>` (for printer use), `iter_segments() -> impl Iterator<Item = (ClassId, VReg, ProgramPoint, ProgramPoint)>`.
    - Add `vreg_to_class(vreg, point) -> Option<ClassId>` inverse lookup. Implementation: maintain a parallel `BTreeMap<VReg, (ClassId, ProgramPoint, ProgramPoint)>` inverse index updated on every `insert_segment`/`insert_full_range`. Eager maintenance keeps the data structure invariants local and trivial to reason about (the splitter will call `vreg_to_class` often during victim selection).
    - Add `pub(crate) split_generation: u32` field, initialized to 0 in `ClassVRegMap::new`. `apply_plan_to` (Phase 5, Task 5.6) bumps it by 1 whenever it commits splitter output. Consumers that must run AFTER the splitter (e.g. `collect_block_param_vregs_per_block` in Phase 6) use `debug_assert!(class_to_vreg.split_generation > 0)` to detect incorrect ordering.
    - Deprecate `lookup_single`: internally call `lookup_any` and emit a `#[deprecated]` warning. Keep until end of Phase 7.
  - Initial population: linearization code in `src/compile/mod.rs:331-492` calls `insert_full_range` so all existing behavior is preserved (the range is the whole function).
  - Every consumer site from the inventory updates its call from `.lookup_single(class)` to `.lookup(class, point)` where `point` is the program point relevant to that consumer. The list below is exhaustive; each site gets the commit note of the program point used:
    - barrier.rs (build_barrier_maps): barrier's barrier-result position.
    - barrier.rs (populate_effectful_operands): barrier position.
    - cfg.rs (collect_phi_source_vregs, compute_copy_pairs): `BLOCK_EXIT(source_block)` for args, `BLOCK_ENTRY(target_block)` for params.
    - effectful.rs: barrier position of the effectful op.
    - precolor.rs: barrier position (Call) or instruction position (Shift/Div).
    - terminator.rs (`build_phi_copies`, `lower_terminator`): `BLOCK_EXIT(current_block)` for Ret/Jump/Branch args; `BLOCK_ENTRY(target_block)` for block params.
    - mod.rs consumers: pass block-entry/exit as documented, with each barrier's specific point where applicable.
    - global_liveness.rs: block-edge points (entry for params, exit for phi_uses args).
    - ir_print.rs, ir/print.rs: use `lookup_any`; the printer intentionally doesn't know program points.
- Test milestone:
  - 636 + 7 new unit tests in `src/egraph/extract.rs` = 643:
    - `multi_segment_insert_and_lookup`
    - `lookup_respects_range_boundaries`
    - `overlapping_segments_reject_via_debug_assert` (debug_assert panics on overlap)
    - `empty_class_returns_none`
    - `lookup_any_picks_first_segment`
    - `iter_segments_yields_all`
    - `truncate_segment_start_updates_inverse` (after truncating a segment, `lookup` at a point below `new_start` returns `None` and `vreg_to_class` at that point also returns `None`; at a point at/after `new_start` both return the expected value)
  - 357 lit tests still pass because all inserts are full-range.
- Risk callout: if a consumer chooses a wrong point (e.g. block-entry when it should be barrier-N), the lookup may return the wrong segment once real splits exist (Phase 5+). Today it still works because each class has one segment. Mitigation: during Phase 4, ADD an assertion in `lookup`: when in `cfg(debug_assertions)`, if the class has >1 segment and more than one covers the point, panic. Phase 5's splitter and its test case will exercise this.
- Dependencies: Phases 1-3.

- [ ] Task 4.1: Rewrite `ClassVRegMap` to segment-vector storage in `src/egraph/extract.rs` (Complexity: Medium)
- [ ] Task 4.2: Add `lookup(class, point)`, `insert_segment`, `insert_full_range`, `iter_segments`, `lookup_any`, `vreg_to_class(vreg, point)` inverse lookup with eagerly-maintained index, `truncate_segment_start(vreg, new_start)` (updates forward `segments` and inverse `vreg_to_class` index atomically), plus debug overlap assertion. Also add a `pub(crate) split_generation: u32` field to `ClassVRegMap`, initialized to 0; bumped by `apply_plan_to` in Phase 5 (Task 5.6) to signal that splitter output has been committed (Complexity: Medium)
- [ ] Task 4.3: Update construction sites in `src/compile/mod.rs` to call `insert_full_range` (Complexity: Low)
- [ ] Task 4.4: Convert `barrier.rs`, `cfg.rs`, `effectful.rs`, `precolor.rs`, `ir_print.rs` consumers to `lookup(class, point)` with the documented points (Complexity: High)
- [ ] Task 4.5: Convert `terminator.rs` and `global_liveness.rs` consumers (Complexity: Medium)
- [ ] Task 4.6: Convert remaining `src/compile/mod.rs` consumers (32 call sites) (Complexity: High)
- [ ] **Mid-phase checkpoint**: confirm `cargo check` compiles after 4.1-4.6. Do not proceed to tests until all call sites updated.
- [ ] Task 4.7: Add 7 new unit tests for multi-segment behavior (listed in the test milestone above) (Complexity: Medium)
- [ ] Task 4.8: Run `cargo fmt`, `cargo clippy`, `cargo test`, lit suite — green baseline (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 4 complete** — segment-keyed map lives with inverse lookup, every consumer passes a point, full-range inserts preserve all 357 lit + 643 unit tests (627 + 1 Phase 1 + 3 Phase 2 + 5 Phase 3 + 7 Phase 4 = 643).

### Phase 5: Pressure-splitter pass (skeleton + live-range analysis, behind env flag)

- Goal: create `src/compile/split.rs` containing the pressure analyzer and a first cut of the splitter that runs behind `BLITZ_SPLIT=1`. When disabled (default), the pass is a no-op and all current behavior is unchanged. The pass does not yet remove any rename plumbing; it only prepares the IR.
- Concrete changes:
  - New file `src/compile/split.rs`:
    - `pub struct SplitPlan { pub per_block_new_insts: Vec<Vec<ScheduledInst>>, pub new_segments: Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>, pub new_slot_count: u32 }`
    - ```
      pub fn plan_splits(
          block_schedules: &[Vec<ScheduledInst>],
          class_to_vreg: &ClassVRegMap,
          extraction: &ExtractionResult,
          egraph: &EGraph,
          cost_model: &CostModel,
          global_liveness: &GlobalLiveness,
          gpr_budget: u32,
          xmm_budget: u32,
          next_vreg: u32,
          loop_depths: &BTreeMap<VReg, u32>,
      ) -> SplitPlan
      ```
    - Internal helpers:
      - `compute_local_liveness(block_idx, &block_schedules[block_idx], global_liveness.live_in[block_idx]) -> Vec<BTreeSet<VReg>>` — live-before sets per instruction. Seeded from `global_liveness.live_in[block_idx]` so pass-through VRegs (live-in with no local def/use) are correctly counted toward pressure. Implementation: initialize the scan's running set to `live_in`, then walk backward across `block_schedules[block_idx]` adding uses and removing defs (standard backward liveness), recording the live-before snapshot at each instruction position.
      - `compute_pressure(live_sets, vreg_classes, budget) -> Vec<(ProgramPoint, u32)>` — overshoots.
      - `score_victim(vreg, live_range, defining_op, loop_depth, call_crosses) -> f64` — uses the same cost metric as current `select_spill_candidates_global` (loop-depth-weighted use count / live-range length, with remat bonus). Reuse `LOOP_DEPTH_PENALTY_BASE` (see Task 5.0 for visibility change).
      - `pick_victim(live_at_overshoot, pressure_overshoot) -> Vec<VReg>` — greedy, picks up to `overshoot` victims.
      - `apply_split(block_schedules_mut, victim, split_point, kind)` — for `Remat(op)`: emit a fresh-VReg copy of `op` before the next use, rewrite operands, update `new_segments`. For `SlotSpill`: emit `SpillStore` at def, `SpillLoad` with fresh VReg before each use, rewrite operands, update `new_segments`. For block-param victims (no explicit def): use the phi-copy-destination-as-slot strategy described in Phase 6.
      - `choose_split_kind(victim, def_site, uses, extraction, egraph, class_to_vreg) -> SplitKind` — resolve `victim`'s ClassId via `class_to_vreg.vreg_to_class(victim, def_site)`, then convert the live-VReg set at the use point to a `BTreeSet<ClassId>` by the same inverse lookup before calling `extract_at`. Remat if the victim's defining op `is_rematerializable` OR `extract_at(class, live_classes_at_use).op.is_rematerializable()`; otherwise `SlotSpill`. Never remats a call-arg VReg (same invariant as today).
    - The pass scans blocks forward, tracks pressure, picks a victim when an overshoot is seen, calls `apply_split`, then re-scans the block. Worst-case bounded by sum of overshoots (finite because each split strictly reduces the surviving victim's live range at that point).
  - Integrate in `src/compile/mod.rs` at line 848 (after `populate_effectful_operands`, before the `allocate_global` call at ~line 870):
    ```
    use crate::regalloc::coloring::{available_gpr_colors, AVAILABLE_XMM_COLORS};
    if std::env::var("BLITZ_SPLIT").ok().as_deref() == Some("1") {
        let gpr_budget = available_gpr_colors(opts.force_frame_pointer);
        let xmm_budget = AVAILABLE_XMM_COLORS;
        let plan = split::plan_splits(
            &block_schedules,
            &class_to_vreg,
            &extraction,
            &egraph,
            &cost_model,
            &global_liveness,
            gpr_budget,
            xmm_budget,
            next_vreg,
            &loop_depths,
        );
        apply_plan_to(&mut block_schedules, &mut class_to_vreg, &mut next_vreg, plan);
    }
    ```
    `global_liveness` is the `GlobalLiveness` produced by `compute_global_liveness_with_block_params` at `compile/mod.rs:~845`; reuse that value rather than recomputing.
  - Tracing: add `BLITZ_DEBUG=split` category (matching existing trace.rs convention).
- Test milestone:
  - 643 + 6 new unit tests in `src/compile/split.rs` = 649:
    - `pressure_overshoot_detected_on_xmm_across_call`
    - `remat_chosen_for_fconst_victim`
    - `slot_chosen_for_noncheap_victim`
    - `split_updates_class_to_vreg_segments`
    - `call_arg_never_remat_even_if_free`
    - `no_overshoot_no_splits`
  - Lit: 357 unchanged. New lit test `tests/lit/regalloc/split_xmm_across_call_shadow.c` — same as `xmm_loop_crossing.c`, but marked `// UNSUPPORTED:` unless `BLITZ_SPLIT=1`. Skipped by default runner. Manually runnable.
  - New lit regression test `tests/lit/regalloc/split_passthrough_xmm.c` — three-block case where an XMM value is defined in block A, used in block C, and block B calls a function without using the XMM (pass-through through B). Must pass with `BLITZ_SPLIT=1`. Document the expected numeric output in an `// OUTPUT:` directive. This directly guards against the local-only liveness bug.
  - `BLITZ_SPLIT=1 bash tests/lit/run_tests.sh`: `xmm_loop_crossing.c` now passes; `split_passthrough_xmm.c` passes; other tests MAY regress — this phase does not promise a green flag-on run yet.
- Risk callout: the greedy victim picker may loop if `apply_split` doesn't strictly reduce pressure at the overshoot point. Mitigation: explicit assertion — after each `apply_split`, re-check pressure at the same point; if not strictly less, panic `split pass failed to reduce pressure`. This is a bug in the picker, not a legitimate outcome.
  - LICM interaction: LICM-hoisted classes enter the splitter as ordinary ClassIds in `class_to_vreg` (LICM adds them as `extra_roots` on the preheader, which gets linearized like any other block). The splitter sees them as normal VRegs; no special case needed.
  - DCE2 interaction: DCE2 runs at `mod.rs:302` before the split pass at `mod.rs:848`; ClassId-level consumption detection is unaffected by splits since splits create new VRegs, not new ClassIds.
- Dependencies: Phases 1-4.

- [ ] Task 5.0: Re-export `LOOP_DEPTH_PENALTY_BASE` as `pub(crate)` in `src/regalloc/spill.rs:10` so `src/compile/split.rs` can reference it. Minimal change; the constant stays where it lives today and is consumed from both sides via `crate::regalloc::spill::LOOP_DEPTH_PENALTY_BASE`. (Complexity: Low)
- [ ] Task 5.1: Create `src/compile/split.rs` and register module (Complexity: Low)
- [ ] Task 5.2: Implement `compute_local_liveness` seeded with `global_liveness.live_in[block_idx]`, `compute_pressure`, `score_victim`, `pick_victim` (Complexity: Medium)
- [ ] Task 5.3: Implement `choose_split_kind` (e-graph-consulting remat decision via `extract_at`, using `ClassVRegMap::vreg_to_class` to map VReg live sets to ClassId live sets) (Complexity: Medium)
- [ ] Task 5.4: Implement `apply_split` for `Remat` and `SlotSpill` paths (block-param path deferred to Phase 6) (Complexity: High)
- [ ] Task 5.5: Implement the outer `plan_splits` loop with pressure-reduction assertion (Complexity: Medium)
- [ ] Task 5.6: Integrate pass behind `BLITZ_SPLIT=1` in `compile/mod.rs`. Import `available_gpr_colors` and `AVAILABLE_XMM_COLORS` from `src/regalloc/coloring` at the call site and pass `available_gpr_colors(opts.force_frame_pointer)` and `AVAILABLE_XMM_COLORS` to `plan_splits`. Pass the existing `GlobalLiveness` from `compute_global_liveness_with_block_params`. Implement `apply_plan_to(block_schedules, class_to_vreg, next_vreg, plan)` in `src/compile/split.rs`; it installs `plan.new_segments` into `class_to_vreg` via `insert_segment`, patches per-block schedules, advances `next_vreg`, and as its final step executes `class_to_vreg.split_generation += 1` to mark that splitter output has been committed. No change to default path. (Complexity: Low)
- [ ] Task 5.7: Add 6 unit tests listed above (Complexity: Medium)
- [ ] Task 5.8: Add lit tests `tests/lit/regalloc/split_xmm_across_call_shadow.c` and `tests/lit/regalloc/split_passthrough_xmm.c`. The pass-through test has block A define an XMM value, block B make a call without using the XMM, block C use it; assert correct numeric output under `BLITZ_SPLIT=1`. (Complexity: Medium)
- [ ] Task 5.9: Run default `cargo test` + lit — all 649 unit / 357 lit pass (flag off). `BLITZ_SPLIT=1` runs: `xmm_loop_crossing.c` and `split_passthrough_xmm.c` both pass; other tests may still regress per Phase 7 caveat. (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 5 complete** — `split.rs` exists with 6 passing unit tests (baseline 649 unit), `split_passthrough_xmm.c` guards pass-through liveness, default pipeline unchanged, behind env flag the splitter can rewrite XMM-across-call IR correctly.

### Phase 6: Block-param split strategy

- Goal: decide and implement how a block param VReg that gets picked as a split victim is handled, since it has no explicit def instruction.
- Recommendation: **stack-destination phi copies in the predecessor** (approach B below). Reasoning:
  - A block param's "def" is the phi-copy write at block entry. If we want the value in a slot across a long live range, the cheapest place to put it in the slot is right at the copy itself, skipping the register round-trip.
  - Approach A (short-lived register + immediate `SpillStore` at block entry) requires the block-param to transiently hold a register the allocator could reuse within the first instruction, which re-introduces the phantom-interference avoidance dance we are trying to remove.
  - Approach B makes the split victim's segment start at `BLOCK_ENTRY(target)`. The reload VReg is defined by a `SpillLoad` inserted at the first use site. No register is allocated to the block-param itself; interference contribution is zero.
- Architecture note (`PhiCopy` emit path): `src/emit/phi_elim.rs::phi_copies` takes `&[(Reg, Reg, OpSize)]` — physical registers only. Rather than change that signature, `lower_terminator` emits the slot store directly. For each `PhiCopy::ToSlot(slot, src_reg, size)` entry, `lower_terminator` emits a `SpillStore` MachInst directly BEFORE calling `phi_copies`, and the `(Reg, Reg, OpSize)` slice passed to `phi_copies` is filtered to contain only the regular copy entries. `phi_elim::phi_copies` signature stays unchanged.
- Concrete changes:
  - `src/compile/split.rs`:
    - `apply_split` gains a `SlotSpillBlockParam { target_block: usize, param_idx: u32 }` arm: allocates a slot, then for every predecessor Jump/Branch to `target_block`, rewrites the phi-copy source: instead of a `MovRR` into the param reg, record a `PhiCopy::ToSlot` entry. At each use in the target block, insert a `SpillLoad slot -> fresh_vreg` and rewrite operand. Update `class_to_vreg` segment.
    - The block-param VReg is NOT in `block_schedules` (block-params are not scheduled instructions; they come from `collect_block_param_vregs_per_block` walking the e-graph). Its ClassVRegMap segment is truncated to start after `BLOCK_ENTRY(target)`; since `collect_block_param_vregs_per_block` performs the lookup at block entry, the param naturally drops out. There is no explicit "delete from schedule" step.
    - Ordering constraint: `apply_plan_to` (which commits new segments to `class_to_vreg` and bumps `class_to_vreg.split_generation` — the `pub(crate) split_generation: u32` field defined in Task 4.2) MUST run BEFORE `collect_block_param_vregs_per_block` in `mod.rs`. Task 6.3 adds `debug_assert!(class_to_vreg.split_generation > 0)` at the top of `collect_block_param_vregs_per_block` to detect incorrect ordering.
  - `src/regalloc/global_liveness.rs`: no change needed — `collect_block_param_vregs_per_block` observes the truncated segment at `BLOCK_ENTRY(target)` automatically.
  - `src/compile/terminator.rs` phi-copy emission learns a new `PhiCopy` enum alongside the existing `(src_reg, dst_reg, size)` tuple. Variants: `Reg(Reg, Reg, OpSize)` and `Slot(Reg, u32, OpSize)`. `lower_terminator` iterates the `PhiCopy` list: for each `Slot` variant, emit a `SpillStore` MachInst directly; for `Reg` variants, collect into a `Vec<(Reg, Reg, OpSize)>` that is passed to `phi_elim::phi_copies` unchanged.
- Test milestone:
  - 2 new lit tests:
    - `tests/lit/regalloc/split_blockparam_across_call.c` — loop counter is XMM, gets split at header param.
    - `tests/lit/regalloc/split_blockparam_multi_pred.c` — multi-predecessor merge block with split param.
  - 4 new unit tests in `src/compile/split.rs` = 653 total:
    - `blockparam_split_truncates_segment_at_entry`
    - `blockparam_split_emits_slot_store_in_predecessor`
    - `blockparam_split_emits_slot_load_at_use`
    - `blockparam_split_two_predecessors_both_store`
  - All prior tests still pass with `BLITZ_SPLIT=1`.
- Risk callout: critical edges. If a predecessor has multiple successors and we rewrite its terminator's phi copies for one successor to slot-store, and another successor for the same predecessor also needs phi copies, ordering matters. Mitigation: `phi_copies` already handles the Briggs-style permutation; the Slot path emits `SpillStore` before `phi_copies` runs, and since `SpillStore` writes memory (never reads any phi-copy destination register), it commutes with `Reg` copies. Add an assertion in `lower_terminator` that any `Slot` copy's `src_reg` is not written by a subsequent `Reg` copy; if that ever fails, inject a pre-save via `R11`.
- Dependencies: Phase 5.

- [ ] Task 6.1: Add `PhiCopy` enum in `src/compile/terminator.rs` with `Reg` and `Slot` variants. Update `lower_terminator` to emit `SpillStore` MachInst for each `Slot` entry before invoking `phi_elim::phi_copies`; the slice passed to `phi_copies` is filtered to `Reg`-variant tuples only. `phi_elim::phi_copies` signature stays unchanged. (Complexity: Medium)
- [ ] Task 6.2: Extend `apply_split` with `SlotSpillBlockParam` branch in `src/compile/split.rs`. Calls `class_to_vreg.truncate_segment_start(victim_vreg, BLOCK_ENTRY(target))` to shrink the victim class's segment so it starts AT the target block entry (the old pre-entry coverage is dropped and the inverse index is kept in sync atomically by the method); emits `PhiCopy::Slot` entries on predecessor terminators; inserts `SpillLoad` at each use site. (Complexity: High)
- [ ] Task 6.3: Add `debug_assert!(class_to_vreg.split_generation > 0, "collect_block_param_vregs_per_block called before splitter committed output")` at the top of `collect_block_param_vregs_per_block` in `src/regalloc/global_liveness.rs`. The field is defined in Task 4.2 and bumped by `apply_plan_to` in Task 5.6. Document the ordering constraint in the function doc comment. (Complexity: Low)
- [ ] Task 6.4: Add 4 unit tests + 2 lit tests listed above (Complexity: Medium)
- [ ] Task 6.5: Run lit with `BLITZ_SPLIT=1` — ensures new lit tests pass AND `xmm_loop_crossing.c` still works (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 6 complete** — Block-param splits produce slot-destination phi copies via the `PhiCopy::Slot` path, baseline is 653 unit tests, new lit tests pass, flagged run handles the motivating regression.

#### Implementation deviation: Task 6.2 `SlotSpillBlockParam` dispatch

The implementation diverges from the original Task 6.2 design. The plan specified routing block-param splitting through the main `SplitKind` enum with a `SlotSpillBlockParam { target_block, param_idx }` variant dispatched by `apply_split_planned`. In practice, block-param VRegs have no scheduled def instruction — they are not in `block_schedules` at all, so the pressure-victim-picker (which operates on scheduled VRegs) cannot select them. Reconstructing their predecessor-edge context would require threading func/egraph state through the picker in a way that is architecturally awkward.

Instead, a dedicated `detect_blockparam_call_crossings` function handles block-param crossing detection directly. It walks the function's block params, checks whether the param VReg is live-in to any block containing a call, and emits `XmmSpillLoad` insertions and `slot_spilled_params` entries in one pass. This is equivalent functionality with clearer separation of concerns. The `SlotSpillBlockParam` variant and `SplitAction` struct were removed from the implementation because they were never constructed. The four Phase 6 unit tests (`blockparam_split_truncates_segment_at_entry`, `blockparam_split_emits_slot_store_in_predecessor`, `blockparam_split_emits_slot_load_at_use`, `blockparam_split_two_predecessors_both_store`) exercise `detect_blockparam_call_crossings` directly and pass.

### Phase 7 (HIGH RISK — parallel-run then cutover)

- Goal: flip the default to `BLITZ_SPLIT=1`, run BOTH old and new paths in the same binary for one commit, then delete the old rename plumbing in a follow-up commit.
- Rationale for split commit: the old plumbing (`coalesce_aliases`, `per_block_rename_maps`, `vreg_slot`, `vreg_remat_op`, spill loop, rebuild_interference, etc.) is scattered across ~1400 lines. Deleting it in the same commit as the default-flip makes rollback painful. Two commits:
  1. Flip default to splitter-on; leave old paths reachable via `BLITZ_SPLIT=0`. Verify 653 unit + 358 lit (now including `xmm_loop_crossing.c`) pass.
  2. Delete old paths. Port deleted function tests first, then delete. Verify final 653 unit + 358 lit.
- Concrete changes (Commit 7A — default flip):
  - `src/compile/mod.rs`: change the env guard from `Some("1")` to `!= Some("0")`. Run splitter unconditionally unless opt-out.
  - Retain `insert_spills_global`, `select_spill_candidates_global`, `select_spill_by_phantom_interference`, `run_phase5`'s spill loop, `rebuild_interference`, `coalesce_aliases`, `per_block_rename_maps`, `vreg_slot`, `vreg_remat_op`, and the alias-chase loops in `terminator.rs:418-423` and `mod.rs:1043-1053` — but now they act on the splitter's output, which should already be feasible, so they become a safety net that usually no-ops.
- Fallback gating (mechanical, not time-based):
  - The fallback trigger is mechanical. In debug builds (`cfg(debug_assertions)`), any activation of the fallback invokes `panic!("split pass produced infeasible IR for function {name}")`. In release builds, the trigger increments a `static AtomicU64 FALLBACK_COUNT` and emits `tracing::error!` with the function name and the allocator error; the fallback then runs the legacy spill loop so the build still succeeds.
  - Task 7.10 (deletion of fallback) is blocked on: `FALLBACK_COUNT.load(Relaxed) == 0` after running the FULL lit suite + `cargo test` with `BLITZ_SPLIT` at its default (on). No `tracing::error!` from the fallback path must have been emitted. There is no "one week" wait; the gate is purely these signals.
- Concrete changes (Commit 7B — cutover):
  - `src/regalloc/mod.rs`: delete fields `per_block_rename_maps`, `vreg_slot`, `vreg_remat_op`, `coalesce_aliases` from `GlobalRegAllocResult`.
  - `src/regalloc/global_allocator.rs`: delete `insert_spills_global` (~180 lines), `select_spill_candidates_global` (~150), `select_spill_by_phantom_interference` (~200), `rebuild_interference`'s spill-loop interactions (specifically the call at `global_allocator.rs:1797`), the entire spill-round loop body in `run_phase5` (keep only "color once, map colors to regs" path). Result: `run_phase5` becomes ~200 lines. The ~15 unit tests living inside or exercising these deleted functions (starting around `global_allocator.rs:2481`) are deleted here; Task 7.4-pre ports their behavioral coverage first.
  - `src/regalloc/split.rs` (old per-block splitter): delete the whole file. Remove `pub mod split;` from `src/regalloc/mod.rs`.
  - `src/regalloc/global_liveness.rs`: delete `apply_block_param_overrides_to_phi_uses` and its call in `compile/mod.rs:~764`. Inside `compute_global_liveness_with_block_params` (`src/regalloc/global_allocator.rs:~836-850`), remove the `renamed_block_param_vregs` / `renamed_phi_uses` construction (the alias-renaming dance). Retain the `block_param_vregs_per_block` parameter — it is still needed for interference seeding; only the renaming logic inside goes away. The function's public signature does not change. Call sites at `global_allocator.rs:845`, `:1815`, `:2406` keep their current argument list. Cross-reference: `rebuild_interference` deletion happens in Task 7.5.
  - `src/compile/mod.rs:1005-1055`: delete the `block_class_to_vreg` construction. Replace every downstream use with direct `class_to_vreg.lookup(canon, ProgramPoint::<the obvious point>)`. Specifically, the block snapshot + rename-map merge + alias chase is gone.
  - `src/compile/terminator.rs`: delete `coalesce_aliases` parameter from `lower_terminator`, `build_phi_copies`. Delete the alias-chase loop at `:418-423`.
  - `src/compile/mod.rs`: remove `block_class_to_vreg_snapshot`, `block_rename_maps`, `coalesce_aliases` locals. Also remove the `apply_block_param_overrides_to_phi_uses` call around line 764. The splitter's output segments ARE the authoritative `class_to_vreg`.
  - Delete safety-net debug asserts at `compile/mod.rs:1219-1248` (they can no longer fire; the splitter guarantees vreg_to_reg coverage).
- Test milestone:
  - Commit 7A: 653 unit + 358 lit (including `xmm_loop_crossing.c`) pass with default flags. `BLITZ_SPLIT=0` also green.
  - Commit 7B: final 653 unit + 358 lit pass. Unit test count reflects: 653 after Phase 6, plus 15 ported tests (Task 7.4-pre) = 668, minus 15 deleted tests inside `global_allocator.rs` and `regalloc/split.rs` (Task 7.5 and 7.7) = 653. Note: the 15 ported tests in Task 7.4-pre REPLACE the 15 deleted — they are equivalent behavioral coverage, counted once. Final: 627 + 1 (Phase 1) + 3 (Phase 2) + 5 (Phase 3) + 7 (Phase 4) + 6 (Phase 5) + 4 (Phase 6) + 15 ported (7.4-pre) − 15 deleted (7B) = **653 unit tests** end-state.
- Risk callout: this is the most dangerous phase. Failure mode A: splitter missed an infeasibility and the single-pass allocator fails. Mitigation: in Commit 7A, if `allocate_global` returns a coloring error while `BLITZ_SPLIT=1` is active, debug builds panic and release builds log `tracing::error!` and increment `FALLBACK_COUNT`, then run the legacy spill loop. Task 7.10 removes the fallback only once `FALLBACK_COUNT == 0` after a full test+lit run. Failure mode B: a consumer's program-point choice was wrong but hidden behind the single-segment invariant. Mitigation: the Phase 4 debug assertion in `lookup` fires when overlap is absent but the point falls outside any segment; we'll see this as a clean error, not a miscompile.
- Dependencies: Phases 1-6.

- [ ] Task 7.1 (Commit 7A): Flip default to `BLITZ_SPLIT=1`, retain legacy paths. Run full test suite including `xmm_loop_crossing.c`. (Complexity: Low)
- [ ] Task 7.2 (Commit 7A): Add fallback-on-allocator-error code path in `allocate_global` caller in `compile/mod.rs`. `cfg(debug_assertions)` branch panics; release branch increments a `static AtomicU64 FALLBACK_COUNT` and emits `tracing::error!` with the function name and the coloring error, then falls back to the legacy spill loop. (Complexity: Medium)
- [ ] Task 7.3 (Commit 7A): Full `cargo test` + lit with `BLITZ_SPLIT=1` (default) and `BLITZ_SPLIT=0`. Confirm `FALLBACK_COUNT == 0` and no `tracing::error!` from the fallback path. Commit. (Complexity: Low)
- [ ] **Intra-phase checkpoint between 7A and 7B**: confirm `FALLBACK_COUNT == 0` after a full test+lit run at the default (splitter on) AND a full test+lit run at `BLITZ_SPLIT=0`. No `tracing::error!` from the fallback. Do not proceed to 7B until both runs are clean.
- [ ] Task 7.4-pre (Commit 7B): Port the ~15 unit tests from the to-be-deleted functions (`insert_spills_global`, `select_spill_candidates_global`, `select_spill_by_phantom_interference`, `rebuild_interference`, and any tests inside the old `src/regalloc/split.rs`) into equivalent split-pass tests in `src/compile/split.rs`'s test module. Each deleted test's behavior (what input produces what spill/remat/split decision) gets a corresponding test exercising the new code path. Run `cargo test --lib` before proceeding to deletion tasks; count must be ≥ 668 (pre-deletion). (Complexity: High)
- [ ] Task 7.4 (Commit 7B): Delete `coalesce_aliases`, `per_block_rename_maps`, `vreg_slot`, `vreg_remat_op` fields from `GlobalRegAllocResult`. Update all read sites. (Complexity: Medium)
- [ ] Task 7.5 (Commit 7B): Delete `insert_spills_global`, `select_spill_candidates_global`, `select_spill_by_phantom_interference`, `rebuild_interference` (at `global_allocator.rs:1797`), and the spill-round loop body in `run_phase5`. Keep the single-pass color-and-map logic. The ~15 tests inside these functions (starting around `global_allocator.rs:2481`) are deleted here — their behavioral equivalents are already in place from Task 7.4-pre. (Complexity: High)
- [ ] Task 7.6 (Commit 7B): Delete the alias-renaming dance inside `compute_global_liveness_with_block_params` (remove `renamed_block_param_vregs` / `renamed_phi_uses` construction at `src/regalloc/global_allocator.rs:~836-850`). Delete `apply_block_param_overrides_to_phi_uses` and remove its call in `compile/mod.rs:~764`. RETAIN the `block_param_vregs_per_block` parameter of `compute_global_liveness_with_block_params` — still required for interference seeding; the function's public signature (and its 3 callers at `:845`, `:1815`, `:2406`) are unchanged. Cross-reference: `rebuild_interference` deletion is in Task 7.5. (Complexity: Medium)
- [ ] Task 7.7 (Commit 7B): Delete `src/regalloc/split.rs` and its `pub mod` declaration. (Complexity: Low)
- [ ] Task 7.8 (Commit 7B): Delete `block_class_to_vreg` construction (`compile/mod.rs:1005-1055`) and switch downstream consumers to `lookup(class, point)`. Delete `block_class_to_vreg_snapshot`, `block_rename_maps`, `coalesce_aliases` locals. (Complexity: High)
- [ ] Task 7.9 (Commit 7B): Delete `coalesce_aliases` parameter from `lower_terminator`, `build_phi_copies`; delete the alias-chase loop in `terminator.rs:418-423`. (Complexity: Medium)
- [ ] Task 7.10 (Commit 7B): Pre-condition — run full lit suite + `cargo test` with `BLITZ_SPLIT` at its default (on). Verify `FALLBACK_COUNT == 0` reads zero and no `tracing::error!` from the fallback path was emitted during the run. If any `tracing::error!` appears, this task is BLOCKED until the root cause is fixed. When clean, delete the `BLITZ_SPLIT` env guard, delete the fallback path and `FALLBACK_COUNT` from Task 7.2. Splitter is now always on. Delete safety-net debug asserts at `compile/mod.rs:1219-1248`. (Complexity: Low)
- [ ] Task 7.11 (Commit 7B): `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test` (expect 653 unit), lit (expect 358) — green. Commit. (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 7 complete** — Old rename plumbing gone; `rg coalesce_aliases src/`, `rg per_block_rename_maps src/`, `rg vreg_slot src/`, `rg vreg_remat_op src/` all return zero hits. 653 unit + 358 lit tests pass including `xmm_loop_crossing.c`. `FALLBACK_COUNT` symbol no longer exists in the tree.

### Phase 8: Cleanup and documentation

- Goal: shake out leftover code, update docs, ensure test coverage of the three split kinds.
- Concrete changes:
  - Delete `lookup_single`/`lookup_any` deprecated paths if not needed (Phase 7 should have replaced them all).
  - Update `CLAUDE.md` pipeline diagram to include "Pressure-driven splitting" between scheduling and regalloc.
  - Cross-link `docs/egraph-reference.md` <-> `docs/split-pass.md` and note `extract_at` in both. The `extract_at` API reference itself was written in Task 3.3; Phase 8.3 only adds cross-references between the two docs.
  - Update `docs/egraph-optimization-roadmap.md` if it discusses regalloc-side changes.
  - Add a new doc `docs/split-pass.md` (user-facing reference): program-point model, `class_to_vreg` segments, split kinds, env flags, how to interpret `BLITZ_DEBUG=split` trace output.
  - Audit `TODO.md` for any item that the refactor makes obsolete.
- Test milestone:
  - Add 1 lit test per split kind we haven't otherwise covered:
    - `tests/lit/regalloc/split_gpr_across_call.c`
    - `tests/lit/regalloc/split_remat_fconst_at_use.c`
    - `tests/lit/regalloc/split_slot_spill_noncheap.c` (e.g., result of a multi-op chain that does not remat)
  - 653 unit + 361 lit (358 + 3 new) all green.
- Risk callout: docs drift from code. Mitigation: have each added doc link back to the specific source file/function it describes.
- Dependencies: Phase 7.

- [ ] Task 8.1: Update `CLAUDE.md` pipeline diagram (Complexity: Low)
- [ ] Task 8.2: Write `docs/split-pass.md` (Complexity: Medium)
- [ ] Task 8.3: Cross-link `docs/egraph-reference.md` and `docs/split-pass.md` (adds "See also" references in both directions); update `docs/egraph-optimization-roadmap.md` if it discusses regalloc-side changes. API reference content for `extract_at` already lives in `egraph-reference.md` from Task 3.3 and is not duplicated. (Complexity: Low)
- [ ] Task 8.4: Add 3 lit tests (Complexity: Medium)
- [ ] Task 8.5: Run full `cargo test` + lit — green baseline (653 unit + 361 lit) (Complexity: Low)
- [ ] **Checkpoint: Verify Phase 8 complete** — Docs updated, 3 new lit tests added and passing, baseline now 653 unit + 361 lit.

## Edge Cases & Risks

- EFLAGS-typed classes cross-block: the existing code in `compile/mod.rs:369-375` forbids flags from surviving a block boundary. The splitter must NOT attempt to split a flags-typed VReg. Mitigation: `pick_victim` skips any VReg whose class has `Type::Flags` or `Type::Pair(_, Flags)`; `egraph.class(canon).ty` is the source of truth.
- Call-arg VRegs: per `CLAUDE.md`, these are never remat'd. Splitter enforces by checking `collect_call_arg_vregs` and treating any candidate in that set as `SlotSpill` only.
- XMM class: pressure budget is 16, all caller-saved. The motivating test case. The splitter applies splits at each Call in a loop header until pressure drops to 16 at every program point. `xmm_loop_crossing.c` needs `base` spilled to a slot across `scale()`, with a reload for the `acc = acc + scale(base)` use inside the loop and the final `acc = acc + base` use after the loop.
- Pass-through liveness: a VReg live-in to a block with no local def or use (transits the block) still contributes pressure. `compute_local_liveness` MUST seed the backward scan with `GlobalLiveness::live_in[block_idx]` so these VRegs count. A lit regression test `split_passthrough_xmm.c` (Phase 5) guards this.
- Remat at loop headers: if a victim's def is pre-loop, remat-at-use inside the loop emits the remat op once per iteration. For `Fconst`/`Iconst` this is fine; for non-free ops it may be worse than slot spill. The cost function already accounts for this because `choose_split_kind` consults `CostModel.cost(op) * loop_depth_penalty` implicitly through `score_victim`, but make this explicit in `choose_split_kind` by only selecting remat when `CostModel.cost(op) <= 1.0` AND `own_cost * loop_depth_penalty <= SlotStoreLoadCost * 1`. The constants here are `SlotStoreLoadCost = 5.0` (two memory ops at ~2.5 each).
- SSA-ness: after splitting, the IR is NO LONGER SSA (a class has multiple defs in different segments). Consumers that assume SSA (e.g., `build_vreg_types` keyed on VReg) continue to work because each fresh VReg gets its own type entry inserted by `apply_split`. Verified by Phase 6 tests.
- Critical edges: the Phase 6 slot-destination phi copies implicitly split the edge if the predecessor has multiple successors and the target has multiple predecessors. Verified by `split_blockparam_multi_pred.c`.
- `BLITZ_SPLIT_FALLBACK`/`FALLBACK_COUNT`: mechanical safety hatch, not a feature. Removed in Task 7.10 once the counter confirms zero activations across a full test+lit run.
- Program-point choice for `populate_effectful_operands`: the "point" is the barrier's position in the schedule, but the barrier's argument operands are themselves live from earlier in the block. The splitter operates on the schedule AFTER `populate_effectful_operands`, so barrier arg operands already appear as ordinary `ScheduledInst.operands`. This is why Phase 5 places the splitter at `compile/mod.rs:848` rather than before the population step.
- Performance: extra `extract_at` calls (O(splits) not O(all classes)) are cheap because `splits` is small (0 for most functions, a few for XMM-heavy loop bodies). `compute_local_liveness` inside the splitter is O(insts^2 per block worst case, O(insts) typical).

## Non-Goals

- SSA-form splitting (LLVM greedy regalloc style): out of scope. We accept non-SSA output from the splitter; the allocator never requires SSA.
- Inter-procedural splitting: the splitter is per-function. Calls are treated as barriers, not points where splits can move across.
- Rematerialization of non-leaf ops (e.g., an X86Add whose children aren't live): out of scope in the first cut. `choose_split_kind` only picks `Remat` for leaf/free ops (`Iconst`, `Fconst`, `StackAddr`, `GlobalAddr`, `Param`, `BlockParam`). Rematerializing a 2-input op requires both inputs to be live at the use site, which requires multi-step `extract_at` to confirm; add later if needed.
- Coalescing by rename: the allocator's existing coalescing (Briggs-Cooper) still runs; the splitter does NOT emit coalesce hints. Copy pairs produced by phi-copy lowering are already in `copy_pairs`.
- Changing the cost model: we reuse the existing `CostModel` unchanged.
- Rewriting the scheduler: the splitter runs after scheduling and does not reorder.
- Removing the allocator's Briggs-Cooper coalescing logic: we only remove the POST-allocation alias-resolution dance (`coalesce_aliases` field). The in-allocator coalescing that merges copy-related VRegs pre-coloring is preserved.
- Changing the barrier group system.
- Changing how inlining, DCE, or LICM operate.

## Open Questions

None at proposal time. All design decisions are made (remat-vs-slot policy, block-param split strategy, two-commit rollout, env flag name, fallback gating signal). If implementation surfaces a blocker, escalate before coding around it.

## Acceptance Criteria (major milestones)

- End of Phase 4: full-range `ClassVRegMap` in place with `vreg_to_class` inverse lookup, every consumer takes a `ProgramPoint`, baseline 643 unit + 357 lit preserved.
- End of Phase 6: `BLITZ_SPLIT=1 bash tests/lit/run_tests.sh` shows `xmm_loop_crossing.c` and `split_passthrough_xmm.c` passing. Default-flag suite 653 unit + 357 lit unchanged.
- End of Phase 7A: default flag flipped, 653 unit + 358 lit green, legacy paths still available via `BLITZ_SPLIT=0`, `FALLBACK_COUNT == 0`.
- End of Phase 7B: legacy paths deleted, `rg` confirms zero hits for `coalesce_aliases`, `per_block_rename_maps`, `vreg_slot`, `vreg_remat_op` in `src/`. 653 unit + 358 lit green (equivalent behavioral coverage preserved via Task 7.4-pre ports).
- End of Phase 8: 653 unit + 361 lit green, documentation updated.

## Final Audit

- [ ] **Final Audit** — Re-read the entire plan. For each task in every phase, verify the implementation exists in the codebase. Run `rg coalesce_aliases src/`, `rg per_block_rename_maps src/`, `rg vreg_slot src/`, `rg vreg_remat_op src/`, `rg 'pub mod split' src/regalloc/mod.rs`, `rg FALLBACK_COUNT src/` — all must return zero hits. Run `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` (expect 653 unit), `bash tests/lit/run_tests.sh` (expect 361 lit) — all must be green with the splitter enabled by default. Confirm `tests/lit/regalloc/xmm_loop_crossing.c` and `tests/lit/regalloc/split_passthrough_xmm.c` pass. List any gaps; all gaps must be resolved before reporting completion.
