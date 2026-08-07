# Hygiene findings, 2026-08-07

A sweep of `src/` against the conventions in `CLAUDE.md`. Every item below was
confirmed by reading the code; the two marked *(unproven)* are reachable by
construction but were not forced in a test.

Ordered by value, not by effort. The first four are correctness, not style.

## Correctness

### 1. A blanket `allow` switches the lint gate off inside the largest file

`src/regalloc/global_allocator.rs:54` is `#![allow(dead_code, unused_variables)]`,
on 3471 lines. Deleting the line and rebuilding produces **8 warnings**:

```
unused variable: `call_points`                            :826
unused variable: `div_points`                             :826
unused variable: `uses_frame_pointer`                     :1394
fields `per_block_liveness`, `next_vreg` never read        :111  (Phase3State)
fields `color_map`, `callee_saved_used` never read        :1001 (Phase4State)
multiple fields never read                                :1302 (Phase5Context, 8 of 10)
function `compute_overshoot` never used                   :1583
function `compute_overshoot_from_coloring` never used      :1598
```

`cargo clippy --all-targets` reads clean across the workspace because of this
one line, and it covers the file the bug priors rank first. It is also what
hides items 2 and 3.

**Fix:** delete line 54, `cargo fix`, delete what remains. ~85 LOC.

### 2. The cross-round coalesce-alias accumulation is dead, and its own comment says what that costs

`global_allocator.rs:1826-1829` states the reason aliases are accumulated across
rounds: *"only the first round coalesces, so a function needing two rounds would
otherwise report no aliases at all and leave every stale entry pointing at a
VReg that no longer exists."* The accumulated map is stored in
`Phase5Context.alias_map` (`:1865`), which nothing reads; `run_phase5:1412` uses
`phase4.alias_map`, the converging round's map only.

So the failure the comment describes is the current behaviour for any function
that needs two or more spill rounds.

**Fix:** pass the accumulated map into `run_phase5` and use it at `:1412`.
Deleting `Phase5Context` (item 6) forces this.

### 3. `vreg_class_map` classifies by the result, which `CLAUDE.md` says never to do

`global_allocator.rs:2329-2341` derives a VReg's register class from
`is_fp_op()` on its defining op. `CLAUDE.md`: *"Operand register class comes from
`Op::operand_reg_class()`, never `is_fp_op()`"* -- the latter describes the
result, and `cvtsi2sd`, `cvttsd2si`, `ucomisd` and `movq` read the opposite
class from the one they write.

Two consequences. It never yields `RegClass::Flags`, so a flags VReg reads as
GPR; `choose_spill_candidates:2083` builds `of_class` from it, making a flags
value an eligible GPR spill candidate, and `insert_spills_global` is handed the
same map at `:2018`. And operand classes are never forced, so a cross-block
live-in falls back to GPR -- the bug `build_vreg_classes_from_insts:115-127`
documents itself as fixing.

**Fix:** delete it; call `build_vreg_classes_from_all_blocks` at `:1936`,
`:2018`, `:2083`.

### 4. A silent `continue` where the neighbouring case is a hard error

`src/compile/terminator.rs:519,651-665`. When `regalloc.vreg_to_reg` has no
entry for a terminator argument the code continues with no copy, no store and
no error, documented as *"legacy: 'flow through cross-block spill slots'
path"* -- a per-block-allocator behaviour. Ten lines above, the structurally
identical "nothing writes the parameter on this edge" case is a `CompileError`.
A dropped phi copy on a back edge is a non-terminating loop, which `:637-639`
says in as many words.

**Fix:** make it the same `CompileError` as `:625-635`. If the global allocator
can legitimately leave a VReg unassigned, that is what needs naming.

## Duplication

### 5. Two distinct types both named `VRegSet`

`regalloc::interference::VRegSet` (sorted `Vec<u32>`, `usize` keys) and
`regalloc::vregset::VRegSet` (bitset, `VReg` keys). Overlapping API. The
re-export at `regalloc/mod.rs:17` points at the bitset one, which nothing
imports by that path -- every consumer spells out `interference::VRegSet`, and
`compile/split.rs:19-21` imports both and aliases one `BlockLiveSet`.

Both representations are justified by their access patterns. The names are not.

**Fix:** rename `vregset::VRegSet` to `PressureSet`, drop the re-export.

### 6. `Phase5Context` clones seven structures per spill round; `run_phase5` reads two fields

`global_allocator.rs:1300-1322,1855-1866`. `cfg_succs`, `phi_uses`,
`block_param_vregs_per_block`, `param_vregs`, `call_arg_precolors`,
`copy_pairs`, `loop_depths` and `alias_map` are cloned on every round and never
read. **Fix:** delete the struct, pass `func_name: &str`.

### 7. Six implementations of the coalesce-alias chase

Canonical: `coalesce.rs:14-22` (`chase_alias`). Hand-rolled copies at
`global_allocator.rs:161`, `:887`, `:937`, `:1327`, `:1836`. **`:887` and `:937`
have no self-loop break and hang on a self-entry**, and `:937` is a
character-for-character copy of `:887` over a map `:935` creates as a clone of
the other. **Fix:** make the alias maps `BTreeMap<VReg, VReg>` and call
`chase_alias` at all five.

### 8. Four copies of "highest VReg index, plus one"

`global_allocator.rs:1672`, `:1795`, `:2008`, `split.rs:2213`, and `fast.rs:76`
makes five. **Fix:** `next_free_vreg(&[Vec<ScheduledInst>]) -> u32` in
`regalloc/mod.rs`.

### 9. Three copies of `build_barrier_context` + `assign_barrier_groups`

`compile/mod.rs:701`, `:890`, `:929`, same four arguments and the same
`non_term_count() > 0` guard.

### 10. Three copies of the class-to-budget table

`global_allocator.rs:1445`, `:1517`, `:1613`. The first hardcodes
`AVAILABLE_XMM_COLORS` where the others take it as a parameter, so they can
disagree. Two die anyway under item 1.

### 11. `insert_spills` is dead in production

`spill.rs:110-136`; the only callers are three tests. `pub` is why no warning
fires. `CLAUDE.md`'s "four passes spill" is stale -- `SlotOwner` names three.

### 12. `compute_global_liveness` is a forwarding wrapper with one production caller

`global_liveness.rs:32-38`. It also generates a false comment:
`global_allocator.rs:1747-1750` says `run_phase3` "uses plain
`compute_global_liveness` which doesn't know about block params", but
`run_phase3:903` calls the `_with_block_params` form.

## Comments the conventions forbid

`CLAUDE.md` bars comments referencing plans, phases, task items, or the history
of how the code got here.

- **Task numbers**: `global_allocator.rs:229` ("Tasks 2.4, 2.4.5"), `:803`
  ("Tasks 3.3/3.4/3.7"), `:806` ("Tasks 3.5/3.7"), `:1377` ("Tasks 5.2-5.7"),
  `:1782` ("Tasks 2.3, 2.4, 2.4.5").
- **A review transcript in a doc comment**: `global_allocator.rs:1349-1383`,
  *"We audited the phi-copy emission path... **Conclusion**: ... **We do NOT
  need a forced-slot pre-spill step.**"* 35 lines.
- **Pointers to an internal planning doc**: `compile/phi_removal.rs:26` ("step
  2's tier 2 in docs/internal/refactor-roadmap.md"), `compile/cfg.rs:338`
  ("step 4 of docs/internal/refactor-roadmap.md is where they go").
- **Describing a subsystem that no longer exists**: `compile/mod.rs:818-823`
  ("per-block with cross-block live range splitting... Single-block fast
  path"), contradicted 42 lines later by `:864-865`; `compile/mod.rs:1622-1624`
  ("single-block uses the old allocator path"); `compile/lower.rs:204-205`
  (names `lower_insts_with_ret`, which does not exist).
- **An orphaned comment with no code under it**: `compile/mod.rs:844-846`.
- **A misplaced doc comment**: `global_allocator.rs:1454-1468` documents
  `format_overshoot` but sits above `greedy_clique_containing`.

## Shapes the conventions call out by name

- `compile/mod.rs:1630` -- `if func.blocks.len() > 1` around a `debug_assert`,
  justified by "single-block uses the old allocator path with its own
  guarantees". There is no old allocator. Drop the guard.
- `coalesce.rs:210-213` -- *"Cannot happen -- non-interference was checked
  above -- but skip defensively."* If it cannot happen, assert; if it can, the
  claim above is wrong and that is the bug.
- `compile/lower.rs:204-206` -- a guard justified by a duplicate in a function
  that does not exist.
- `compile/licm.rs:27-47` -- `detect_back_edges` takes `rpo` and discards it
  with `let _ = rpo`. Drop the parameter and update the 5 call sites.
- `compile/mod.rs:1603-1611` -- an undocumented `BLITZ_PROBE` env var with a raw
  `eprintln!`, read inside the per-block loop, bypassing `crate::trace`.
- `egraph/extract.rs:772` -- a "cycle guard" whose comment says e-graphs are
  DAGs. The map is memoization; calling it a cycle guard is what makes it look
  defensive.

## Performance, unmeasured

- `global_allocator.rs:1671,1807` -- the whole function's schedules are deep
  cloned twice per spill round.
- `global_allocator.rs:2190-2211` -- the set-cover loop is O(candidates x
  uncovered) per iteration. Only on the spill path; measure first.

## Two that are latent bugs but unproven

- *(unproven)* `compile/mod.rs:1490-1528` -- two loops emit the function-entry
  parameter mov and only the second handles XMM. `precolor.rs:45-47` skips XMM
  params when the entry block has calls, so such a param lands in the first
  loop and would get `MovRR { size: S64 }` on XMM registers. Reachable by
  construction; not forced in a test. The first loop also runs for every block
  despite its comment saying "at the very start of the function".
- *(unproven)* `global_allocator.rs:887,937` -- the two alias chases with no
  self-loop break (item 7) hang rather than misbehave, so a self-entry in the
  alias map is a compiler hang. No input was found that produces one.
