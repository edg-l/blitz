# Function Inlining Plan (Revised)

## Overview

Function inlining pass for the Blitz compiler that operates on the e-graph IR before optimization phases. The pass replaces `EffectfulOp::Call` sites with the callee's body by importing e-graph nodes and CFG blocks into the caller, with correct remapping of ClassIds, BlockIds, stack slots, and UIDs.

## Requirements

- **Explicit**: Inline function calls at the IR level (post-builder, pre-optimization), controlled by heuristics and CompileOptions fields.
- **Explicit**: Avoid hashcons aliasing when importing callee e-nodes into the caller's e-graph.
- **Explicit**: Handle UID collision for LoadResult and CallResult nodes across caller/callee boundaries.
- **Explicit**: Remap BlockParam, StackAddr, and all BlockId references in imported effectful ops.
- **Explicit**: Support multi-pass inlining (inline into previously-inlined code) with depth bounds.
- **Explicit**: Dead function elimination after inlining.
- **Inferred**: Only inline callees with 0 or 1 return values (Ret carries `Option<ClassId>`).
- **Inferred**: Preserve deterministic output (BTreeMap/BTreeSet throughout the inline module).
- **Inferred**: Guard against missing e-graphs, empty param_class_ids, recursive calls.
- **Assumption**: Inlining happens on `Vec<Function>` before `compile_module` dispatches to per-function compilation.
- **Assumption**: Callee functions are not mutated; they are cloned/read during import into each call site.

## Architecture Decision

**Approach**: A standalone `src/inline/` module with three files: `callgraph.rs` (call graph + heuristics), `remap.rs` (e-graph import with ClassId/BlockId/UID remapping in a single ordered walk), and `transform.rs` (CFG surgery: block splicing, return-to-jump conversion, continuation block creation). A top-level `inline_module()` function orchestrates the pass.

**Why single ordered walk for e-graph import**: The original plan had separate "import nodes" and "substitute params" steps, but `egraph.add()` uses a hashcons memo. If we insert `Op::Param(0, I64)` from the callee, it collides with the caller's existing `Param(0, I64)` and returns the caller's ClassId, silently aliasing unrelated values. By never inserting Param nodes and instead pre-seeding the ClassId remap with `callee_param_class_id[i] -> call_arg_class_id[i]`, we avoid the collision entirely and merge param substitution into the import walk.

**Alternatives rejected**:
- *Renumber all callee Param ops with unique indices*: Fragile, breaks the invariant that Param(i) means the i-th function parameter.
- *Separate e-graph per callee, merge after*: EGraph has no merge-graph API; would require building one from scratch.

## Implementation Plan

### Phase 1: Infrastructure and Call Graph

- [ ] Task 1.1: Create directory `src/inline/` and file `src/inline/mod.rs` with `pub mod callgraph; pub mod remap; pub mod transform;` and the public entry point signature: `pub fn inline_module(functions: &mut Vec<Function>, opts: &CompileOptions)`. Body is a placeholder `todo!()` for now. (Complexity: Low)

- [ ] Task 1.2: Add inlining fields to `CompileOptions` in `src/compile/mod.rs`: `pub enable_inlining: bool` (default `false`), `pub max_inline_depth: u32` (default `3`), `pub max_inline_nodes: usize` (default `50`, callee e-graph node count threshold). Update `Default` impl. (Complexity: Low)

- [ ] Task 1.3: Register the `inline` module in `src/lib.rs` with `pub mod inline;`. (Complexity: Low)

- [ ] Task 1.4: Create `src/inline/callgraph.rs`. Implement `pub fn build_call_graph(functions: &[Function]) -> BTreeMap<String, BTreeSet<String>>` that iterates every block of every function, finds `EffectfulOp::Call { func, .. }` ops, and records caller->callee edges. Return type maps caller name to set of callee names. Use BTreeMap/BTreeSet. (Complexity: Low)

- [ ] Task 1.5: In `callgraph.rs`, implement `pub fn should_inline(callee: &Function, depth: u32, opts: &CompileOptions) -> bool`. Conditions that return false: (a) `depth >= opts.max_inline_depth`, (b) `callee.egraph.is_none()`, (c) `callee.return_types.len() > 1`, (d) `callee.egraph.as_ref().unwrap().node_count() > opts.max_inline_nodes`, (e) `callee.param_class_ids.is_empty() && !callee.param_types.is_empty()` (builder did not populate param IDs), (f) callee has no blocks. Otherwise return true. (Complexity: Low)

- [ ] Task 1.6: In `callgraph.rs`, implement `pub fn is_recursive(name: &str, call_graph: &BTreeMap<String, BTreeSet<String>>) -> bool`. Returns true if `name` is reachable from itself via BFS/DFS on the call graph (direct or mutual recursion). Recursive functions are never inlined. (Complexity: Low)

- [ ] **Checkpoint**: `cargo check` must pass. All 6 tasks done.

### Phase 2: E-graph Import and Remapping

- [ ] Task 2.1: Create `src/inline/remap.rs`. Define `RemapContext` struct with fields: `class_map: BTreeMap<ClassId, ClassId>`, `block_map: BTreeMap<BlockId, BlockId>`, `slot_offset: u32`, `uid_offset: u32`. (Complexity: Low)

- [ ] Task 2.2: Implement `RemapContext::new(caller: &Function, callee: &Function, call_args: &[ClassId]) -> Self`. Compute `slot_offset = caller.stack_slots.len() as u32`. Compute `uid_offset` by scanning caller for max existing UID + 1. Compute `block_map`: allocate new BlockIds starting from `caller.blocks.len() as u32`. Seed `class_map` with `callee.param_class_ids[i] -> call_args[i]` for each parameter. (Complexity: Medium)

- [ ] Task 2.3: Implement `fn max_uid_in_function(func: &Function) -> u32`. Iterate all blocks/ops; for Load/Call results, find LoadResult/CallResult UIDs in the e-graph. Return max UID (or 0). (Complexity: Medium)

- [ ] Task 2.4: Implement `RemapContext::import_egraph(&mut self, caller_egraph: &mut EGraph, callee_egraph: &EGraph)`. Single-pass ordered walk: (a) Iterate callee ClassIds 0..N. (b) Skip if already in class_map (params pre-seeded). (c) For each e-node, rewrite Op: `Param(..)` -> skip (must already be mapped; panic if not), `BlockParam(block_id, idx, ty)` -> rewrite block_id via block_map, `LoadResult(uid, ty)` -> offset uid, `CallResult(uid, ty)` -> offset uid, `StackAddr(i)` -> offset by slot_offset, all others -> as-is. (d) Remap children via class_map. (e) `caller_egraph.add(rewritten_enode)` -> new ClassId. (f) Insert into class_map. Debug-assert all children are in class_map before remapping. (Complexity: High)

- [ ] Task 2.5: Implement `RemapContext::remap_class_id(&self, id: ClassId) -> ClassId`. Lookup in class_map, panic if missing. (Complexity: Low)

- [ ] Task 2.6: Implement `RemapContext::remap_effectful_op(&self, op: &EffectfulOp) -> EffectfulOp`. Remap all ClassId/BlockId fields per variant. (Complexity: Medium)

- [ ] Task 2.7: Implement `RemapContext::remap_blocks(&self, callee: &Function) -> Vec<BasicBlock>`. New blocks with remapped IDs and ops. (Complexity: Low)

- [ ] **Checkpoint**: `cargo check` must pass. All 7 tasks done.

### Phase 3: Core Inlining Transform

- [ ] Task 3.1: Create `src/inline/transform.rs`. Implement `pub fn inline_call_site(caller: &mut Function, block_idx: usize, op_idx: usize, callee: &Function, opts: &CompileOptions)`. Extract Call op, assert it's a Call, extract call_args/call_results/ret_tys. (Complexity: Low)

- [ ] Task 3.2: Append callee stack slots to caller. (Complexity: Low)

- [ ] Task 3.3: Construct RemapContext, call import_egraph, call remap_blocks. (Complexity: Low)

- [ ] Task 3.4: Create continuation block. ID = max of all existing + remapped block IDs + 1. If ret_tys.len() == 1: one block param. If empty: no params. (Complexity: Medium)

- [ ] Task 3.5: Split caller block at call site. Ops before Call stay in original block. Ops after Call move to continuation block. Original block gets `Jump { target: remapped_callee_entry, args: [] }` as terminator. (Complexity: Medium)

- [ ] Task 3.6: Rewrite callee Ret ops to Jump to continuation. `Ret { val: Some(v) }` -> `Jump { target: cont_id, args: [v] }`. `Ret { val: None }` -> `Jump { target: cont_id, args: [] }`. (Complexity: Low)

- [ ] Task 3.7: Merge CallResult with continuation block params. If ret_tys.len() == 1: add `BlockParam(cont_id, 0, ret_tys[0])` e-node, merge with call_results[0], call `egraph.rebuild()`. (Complexity: Medium)

- [ ] **Checkpoint**: `cargo check` must pass. All 7 tasks done.

### Phase 4: Module-level Pass and Integration

- [ ] Task 4.1: Implement `inline_module()` body. Assert function name uniqueness. Build call graph. For each function, scan blocks for Call ops. Check `!is_recursive && should_inline`. If yes, clone callee, call inline_call_site. (Complexity: High)

- [ ] Task 4.2: Implement re-scan loop. `while changed` loop, restart from block 0 after each inline. Bound by `max_inline_depth * 10` iterations. Track depth per function. (Complexity: Medium)

- [ ] Task 4.3: Wire into `compile_module()`: call `inline_module` on Vec<Function> before the per-function loop, gated by `opts.enable_inlining`. (Complexity: Low)

- [ ] Task 4.4: Wire into `compile_module_to_ir()`, same pattern. (Complexity: Low)

- [ ] **Checkpoint**: `cargo check` and `cargo test` must pass (inlining off by default).

### Phase 5: Unit Tests

- [ ] Task 5.1: `test_inline_simple_leaf` -- inline callee returning Iconst(42). Verify no Call ops remain, CallResult resolves to 42. (Complexity: Medium)
- [ ] Task 5.2: `test_inline_void_callee` -- void callee with Ret { val: None }. Verify continuation block has no params. (Complexity: Low)
- [ ] Task 5.3: `test_inline_with_stack_slots` -- callee has 2 stack slots. Verify caller.stack_slots grew, StackAddr remapped. (Complexity: Medium)
- [ ] Task 5.4: `test_inline_multi_return_skipped` -- callee with 2 return types. should_inline returns false. (Complexity: Low)
- [ ] Task 5.5: `test_inline_recursive_skipped` -- direct and mutual recursion detected. (Complexity: Low)
- [ ] Task 5.6: `test_inline_depth_limit` -- chain a->b->c with max_depth=1. b inlined, c not. (Complexity: Medium)
- [ ] Task 5.7: `test_callgraph_build` -- 3 functions, verify edges. (Complexity: Low)

- [ ] **Checkpoint**: `cargo test` passes, all 7 tests exist and pass.

### Phase 6: Dead Function Elimination

- [ ] Task 6.1: Implement `fn eliminate_dead_functions(functions: &mut Vec<Function>, entry_names: &BTreeSet<String>)`. BFS reachability from entry set through call graph. Remove unreachable. (Complexity: Medium)
- [ ] Task 6.2: Call at end of `inline_module`. Entry set = `{"main"}`. (Complexity: Low)
- [ ] Task 6.3: Test: leaf called once is eliminated after inlining. (Complexity: Low)
- [ ] Task 6.4: Test: function with remaining call sites is kept. (Complexity: Low)

- [ ] **Checkpoint**: `cargo check` and `cargo test` pass.

### Phase 7: tinyc Integration and E2E Tests

- [ ] Task 7.1: Add `--inline` flag to tinyc CLI. Set `opts.enable_inlining = true`. (Complexity: Low)
- [ ] Task 7.2: Propagate inlining option through tinyc lib compile functions. (Complexity: Low)
- [ ] Task 7.3: Lit test `inline_simple.c` -- leaf function inlined, no Call in IR. (Complexity: Medium)
- [ ] Task 7.4: Lit test `inline_void.c` -- void function inlined, correct output. (Complexity: Medium)
- [ ] Task 7.5: Lit test `inline_recursive_no.c` -- recursive function NOT inlined, correct result. (Complexity: Low)
- [ ] Task 7.6: Run full test suite, verify no regressions. (Complexity: Low)

- [ ] **Checkpoint**: All tests pass.

- [ ] **Final Audit** -- Re-read entire plan, verify each task has implementation, list and resolve gaps.

## Edge Cases and Risks

- **Param hashcons aliasing**: Mitigated by never inserting Param nodes during import; callee param ClassIds are pre-mapped to call argument ClassIds in RemapContext::new.
- **LoadResult/CallResult UID collision**: Mitigated by scanning caller for max UID and offsetting all callee UIDs.
- **BlockParam stale block IDs**: Mitigated by rewriting BlockParam ops during e-graph import with remapped block_id.
- **Multi-return callees**: Guarded by precondition in should_inline (return_types.len() > 1 -> skip).
- **Missing egraph**: Guarded by should_inline checking callee.egraph.is_none().
- **Missing param_class_ids**: Guarded by should_inline checking empty param_class_ids when param_types is non-empty.
- **Recursive/mutual recursion**: Guarded by is_recursive DFS check.
- **Infinite re-scan loop**: Bounded by max_inline_depth * 10 iteration cap.
- **Non-determinism**: All maps use BTreeMap/BTreeSet.
- **ClassId ordering in import_egraph**: Bottom-up construction guarantees children have lower IDs. Debug-assert verifies.
- **rebuild() placement**: Called once in Task 3.7 after all merges.
- **Block ID collision**: block_map allocates from caller.blocks.len() upward; continuation block uses max+1.
