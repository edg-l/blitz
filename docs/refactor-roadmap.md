# Refactor roadmap

The structural work, in the order it should happen and with the reason for that
order. Written to survive across sessions: it assumes no context beyond the repo.

Judge every step on the full battery (`cargo test --all-targets --workspace`,
`bash tests/lit/run_tests.sh` at `BLITZ_VERIFY` off/1/strict,
`bash tests/lit/run_diff.sh`) and per **(seed, level) pair** with
`tests/fuzz/compare_ref.sh <ref> 60 <shape>`, never on a bare pass count.

## Before you start: three cheap things that make the rest easier

None of these changes behaviour. Each one shrinks the surface or the risk of the
steps below, and they are worth doing first rather than carrying through a rewrite.

**A. A CFG-versus-schedule agreement check, under `BLITZ_VERIFY`.** DONE.
`verify::verify_cfg_schedule_agreement` compares, at every position both
representations name, the VReg a class resolves to through the map against the VReg
the schedule carries as an operand: an effectful op's role operands against the
barrier consuming it. Enforced at construction, where it is clean over the whole
suite; reported after the splitter, where it is not. Step 3 makes it vacuous, at
which point it goes.

**The number when it was written, over the 440-test lit suite: 0 disagreements at
construction, 16 programs disagreeing after the splitter.** Two shapes, and neither
was a live bug because neither consumer trusts the map first -- an effectful op reads
the barrier's own operand and falls back to the map only where no operand exists at
that index, and a terminator argument was read off `Op::TerminatorArgs` with no
fallback at all.

- **Two `SpillLoad`s of one class at the same point**, each with a segment of its own
  and both covering it. `ClassVRegMap::lookup` has nothing to choose by and asserts;
  `lookup_ambiguity` exposes that as a value so the check reports it. Also a missed
  CSE: one reload would do.
- **A reload whose segment does not reach the barrier that consumes it**, so the map
  still answers with the pre-spill VReg -- `v864 = SpillLoad(20)` at index 48 is what
  the `StoreBarrier` at index 152 reads, and the map at 152 says `v30 = StackAddr(0)`.
  Segments are keyed on raw instruction indices, and a later split round inserting an
  instruction ahead of one moves the point it was measured against.

Step 1 removed the terminator half of this check outright: block arguments no
longer have two answers to compare. What the check covers now is the effectful-op
role operands alone, and both shapes above are still reachable through those.

Also found and left alone, since it is a behaviour change and this step is not:
`populate_effectful_operands` resolves at `ProgramPoint::barrier_point`, which is the
barrier's index **plus one**, while `lower_effectful_op` resolves at the raw index.
The splitter's segments run `[def .. last reference]` over raw indices, so only the
second lands inside a reload's segment. Harmless today because the first runs before
the splitter, where every segment is full-range.

**B. Delete the stale plan references.** DONE. All 65 `Task N.N` references are gone.
A `Phase N` stays only where the repo defines the number: the pipeline's list in
`compile/mod.rs`'s module doc, and the allocator's own `run_phase3`/`4`/`5`. The
other 116 named the deleted plan's phases and collided with both of those -- "the
Phase 6 splitter" is not the pipeline's phase 6 and "merged away by Phase 3" is not
its phase 3 -- so those now name the pass. Where a comment's whole content was the
task number it states the constraint instead.

**C. One `block_param_map`, built once and threaded.** DONE.
`EGraph::block_param_classes` is the single scan, sitting next to
`rewrite_block_params`, the write counterpart. Six passes had their own copy;
`collect_block_param_vregs_per_block` and `split::find_block_param_vreg` ran theirs
from *inside* a per-parameter loop, so a block with 28 parameters walked every class
28 times to answer 28 questions one pass answers. Both now take the map
`compile/mod.rs` already builds once.

The fourteen parameter-VReg resolutions collapse to
`cfg::resolve_block_param_vreg`, which names the four sources and why the order is
what it is. Two of them were the same four-source chain written twice, in
`compute_copy_pairs_from_schedules` and `build_phi_copies`, with a doc comment on
the first saying it must match the second -- and they had already drifted: one
canonicalized the parameter's class before the lookup and one did not. Coalescing
merging onto a VReg the copy never writes is what `021d4ed` and `29e796d` were.

Not worth doing first: splitting the large files (no evidence of harm), and the `Op`
split -- that is step 7, and doing it early fights steps 1-4, which change what the
CFG holds.

## Order, and why

| # | step | size | why here |
| --- | --- | --- | --- |
| 0 | Slot numbering gets a real representation | small | **DONE** — was a documented landmine, and a prerequisite for step 5 |
| 1 | Terminator args become VRegs in the CFG | medium | **DONE** — the representation defect, smallest slice with the biggest payoff |
| 2 | Trivial-phi elimination over those VRegs | small-medium | 85-94% of every function's parameters; removes 36 of the 46 capacity failures |
| 3 | The remaining `EffectfulOp` operands become VRegs | medium | finishes step 1 |
| 3b | Block parameters get VRegs in the CFG | small | the phi seam's *destination* end; step 4 deletes its four-source resolution and needs a replacement |
| 4 | Delete the reconstruction machinery (~312 sites) | mechanical | the payoff: the bug class goes away |
| 5 | Give the function-scope allocator a spill loop | medium | now safe, and it converts the residual failures from errors into spill code |
| 6 | Fold the two allocators into one | medium | only sensible after 5, or the spill loop gets duplicated |
| 7 | Split `Op` into pure / machine / pseudo | large | lowest priority, natural follow-on to 1-4 |

**Why not the spill loop first.** It is tempting: it is self-contained, the code to
crib from already exists in `allocator.rs`, and it would turn all 46 remaining
capacity failures from compile errors into merely-slow code — a strictly better
baseline to refactor against. But a spiller is *precisely* a pass that mints new
VRegs for existing classes, which is the thing the current representation handles
worst. `CLAUDE.md` states the hazard outright — *"Anything in lowering that
re-resolves a class on its own must account for spills and rematerialization — this
seam produced seven separate wrong-code bugs"* — and the constant-remat pass, which
is the same shape, was implemented and reverted twice on it. Adding a spiller before
step 4 means re-learning those bugs.

Correctness is currently good (no wrong-value program in the corpus at 60 seeds per
shape), so there is no urgency forcing the order the other way. Steps 1-4 are judged
by correctness gates that are green and comprehensive; capacity stays where it is
during them, which is annoying and not a regression.

---

## Steps 1-4: the CFG should hold VRegs, not ClassIds

### Root cause

**The CFG names pure values by `ClassId` — expression identity — while every
consumer after extraction needs *occurrence* identity: which register holds this
value, in this block, at this point.**

An e-class has no position. A block parameter fundamentally does, and blitz stores
`Op::BlockParam(block, idx, ty)` as an e-graph node, which makes a
position-dependent value look like a position-free one. So the back half of the
pipeline *reconstructs* position from a function-wide map, and every reconstruction
is a place to get it wrong.

The evidence, all of it already in the repo before this document:

- `CLAUDE.md`, key design decisions: *"An effectful op's operands are `ClassId`s in
  the CFG, not VRegs in the schedule, so the splitter's operand rewriting never
  reaches them… this seam produced seven separate wrong-code bugs."*
- Same file: *"One e-class can map to several VRegs, and which one is right depends
  on the block and the program point… Resolving a per-block question through the
  function-wide map, or through a snapshot taken before the splitter ran, is the
  single most common source of wrong-code bugs here."*
- Every wrong-code fix of the 2026-08-03 sessions replaced a class lookup with the
  VReg the schedule carries: terminator arguments (`f5ab27a`), phi destinations
  (`29e796d`), coalescing's copy pairs (`021d4ed`), a division's projections, a
  call's result (`8472e14`), a load's destination, a `Ret`'s value, and a compare's
  operand width (`9207141`).
- The constant-remat pass was implemented, measured and reverted **twice**, both
  times on this seam and never on its policy.
- Redundant block-parameter elimination (`src/compile/phi_simplify.rs`) is blocked
  on the same thing one level up: **one e-class is one expression, not one value**,
  so two predecessors computing the same expression share a class while each emits
  it into a register of its own — which is exactly what the phi reconciles.
- The size of the reconstruction machinery: **224** references to `class_to_vreg`
  and the per-block snapshots, **88** to `block_param_vregs`,
  `block_param_vreg_overrides` and `class_emitted_in`.

Chasing the splitter or the colourer is treating symptoms. Three attempts were
measured on 2026-08-03 — a Chaitin-Briggs simplify/select colourer, cross-round
victim dedup, and a routing-target change — and all were neutral or worse. They are
recorded under "Ruled out" in `docs/terminator-args-next-steps.md`.

### The fix

Extraction already decides, per block, which VReg carries each class. **Commit that
decision into the IR instead of deferring it to Phase 7.** The CFG becomes
VReg-based immediately after extraction, and class → VReg resolution happens in
exactly one place, once.

Then trivial-phi elimination is a small pass whose *operands* are unambiguous,
because a committed argument is a def rather than an expression. Its destinations
are not made unambiguous by that alone: removing a parameter leaves the block
naming a class no single predecessor's emitter dominates, so the removal has to
re-run linearization rather than patch it, and the target block re-emits the class.
That turns the question blocking the pass today into an explicit cost-model
choice — cheap for a constant or an address, and `src/egraph/cost.rs` already
exists to decide. It is the same mechanism the reverted remat pass needed. See
"Removal re-runs linearization" under the step 2 notes.

### What it buys, in order of size

1. **85-94% of block parameters go**, with their copies, their cliques, their slot
   routing, and their per-iteration store/reload traffic.
2. **Rematerialization becomes expressible** — the second big lever for
   pressure-bound code, and the reason two attempts at it failed.
3. The splitter's operand rewriting reaches everything, so pressure decisions stick.
4. The bug class that has consumed most sessions disappears, so optimization work
   stops being interrupted by miscompiles.

### Step 1 — terminator args — DONE

`EffectfulOp::Jump.args`, `Branch.true_args` and `Branch.false_args` are a
`TermArgs`: `Classes(Vec<ClassId>)` while the IR is still being transformed,
`Committed(Vec<TermArg>)` from `cfg::commit_terminator_arg_vregs` onward. One
discriminant per argument list, so a half-committed list is unrepresentable, and
the passes that run either side of the commit say which they expect
(`expect_classes`, `as_committed`) instead of agreeing by convention.

The commit runs at the end of linearization — after extraction and DCE2, before
scheduling and the splitter — and is the *only* place an argument class is
resolved to a VReg. It carries over the two answers `append_terminator_args` used
to consult: the block's own snapshot, and the parameter overrides for a
freshly-minted parameter VReg or a back edge passing a parameter straight
through.

**A committed argument carries both its VReg and its `ClassId`,** because they
answer different questions. `build_phi_copies` asks the class exactly once, to
tell an argument that *is* the parameter it feeds (a loop-carried value the slot
already holds) from one that merely equals it; no VReg comparison expresses that,
and nothing else reconstructs a VReg from the class.

What followed from it:

- `append_terminator_args` copies the CFG's list into `Op::TerminatorArgs` rather
  than resolving it, so the two cannot disagree and the agreement check lost its
  terminator half along with `param_override_vregs_per_block`.
- `collect_phi_source_vregs` no longer takes the e-graph or the class map at all,
  and `compute_copy_pairs`'s argument side is a field read. Both are the
  single-block path.
- `terminator_arg_classes` is gone; `terminator_arg_count` is what the argument
  numbering needs, and `terminator_edges` hands out the `TermArgs` itself.

`Committed` is deliberately not maintained past the splitter: slot routing takes
an argument out of the register file, which a list of VRegs cannot express.
`Op::TerminatorArgs` stays the post-split carrier because its operands are tagged
with argument indices and so can be missing one.

Judged by byte identity, since it changes no behaviour: 854 emitted-asm
comparisons against the pre-change compiler (440 lit programs and 90 generated
ones at both levels) are **identical, with 0 differing and 0 differing in
status**. 934 unit, 440 lit at `BLITZ_VERIFY` off/1/strict, 281 differential +
`cc`. Capacity is unchanged, as expected; the 36 shape-B failures are step 2's
prize.

Still on classes, and left alone on purpose: `Ret.val`, because lowering reads it
to materialize a constant return straight into the ABI register, and the
destination end — `cfg::resolve_block_param_vreg` still answers "which VReg is
this parameter" from four places. Step 4 deletes those; step 2 will ask that
chain for its self-reference test.

- `EGraph::rewrite_block_params(keep)` is already written and is the reusable half
  of removing parameter positions: it renumbers `BlockParam` nodes in one pass over
  a drained memo, because the new index of a surviving parameter is usually the old
  index of a removed one and any incremental rewrite collides.

### Step 2 notes — trivial-phi elimination

`phi(v, …, v) → v`, self-references ignored, iterated to a fixpoint.

- The self-reference rule is what reaches loops: a value carried round a loop and
  never reassigned is `p = phi(p_init, p)`, whose operands reduce to `{p_init}`.
  Without it nothing in a loop is ever removable.
- Do **not** re-attempt it on `ClassId`s. That is what `src/compile/phi_simplify.rs`
  does and its module doc records both failures: removing the parameter reads a
  register only one path wrote (the machine verifier says so outright), and adding
  the sound dominance condition is too strong to win anything *and* still leaves
  programs wrong. That module is superseded by this step, not fixable in place.
- The entry block's parameters are the function's own. Never remove them.
- `tests/fuzz/count_trivial_phis.py` measures the prize off `--emit-ir` and needs no
  compiler change, so the number is checkable before and after.

#### Removal re-runs linearization; it does not patch it

**Analyse on the committed VRegs, remove on the CFG, then linearize again.** The
pass reads each block's snapshot *before* `commit_terminator_arg_vregs` runs, so it
mutates nothing; where it finds anything to remove it drops those parameters while
the argument lists are still `TermArgs::Classes`, and linearization runs a second
time over the reduced CFG.

**This is what makes the transform sound, not a tidier way to arrange it.** A block
parameter exists to reconcile two predecessors that computed the same *expression*
into different *registers* — one e-class, two values. Remove it and the block names
that class directly, but neither emitter dominates the block, so on some path
nothing has written the register it reads. What supplies the missing definition is
linearization's dominance filter re-emitting the class in the block. There is
nowhere else to put it: that is why the two `phi_simplify.rs` attempts either
miscompiled or needed a dominance condition strong enough to win nothing back
(`pressure` seed 22 returned to `gpr_overshoot=4`).

**That re-emission is lever 2, not a cost.** Trading a copy per incoming edge per
iteration plus a place in the parameter clique for one recomputation in the target
block is a win whenever the class is cheap to recompute — an `Iconst`, a
`StackAddr`, a `lea`, which is most of what these parameters carry. Re-emission is
linearization's decision and `src/egraph/cost.rs` is already there, so the choice
has a home for the first time. An in-place patch cannot consult a cost model
because it never makes the choice.

**One dial, with a derived default.** Remove a parameter when the extraction cost of
its class is below the copies it saves — one per incoming edge, weighted by loop
depth from `compute_loop_depths` exactly as the splitter weights spills — plus the
clique slot. Keep it otherwise.

**What patching in place would forfeit,** beyond being unsound: the classes that
were re-emitted per block *because* a parameter mediated them stay re-emitted, so
the redundant materializations remain; the schedule keeps the order it had with the
markers in it, so the liveness the splitter measures no longer matches the code
emitted — the second of the two bug shapes `CLAUDE.md` names; and coalescing sees a
copy set containing copies that should not exist. Any patch thorough enough to fix
those *is* the second linearization.

It would also be the larger job. Everything keyed on `(BlockId, param_index)` would
need patching in step with the removal: `block_param_vregs` (71 references),
`slot_spilled_params` (50), `block_param_map` (48), `block_param_vregs_per_block`
(38), `block_param_vreg_overrides` (16), plus `param_types` (77) and the
`Op::BlockParam` nodes (44). Re-linearizing rebuilds all of it, and
`EGraph::rewrite_block_params` already renumbers the nodes. Which is why the size
in the table is small-medium rather than the "small" this step was first written as:
small as a pass, medium as a pipeline change.

**Cost.** One extra linearization, paid only when there is something to remove. It
then runs over an IR with 85-94% fewer parameters, as do the scheduler and the
splitter — and the splitter is 72% of compile time. Expect net faster; measure it
with `tests/profile.sh` rather than asserting it.

**Judged by `compare_ref.sh` per (seed, level) pair on all three shapes**, plus the
correctness battery. `run_identity.sh` is the wrong gate here: this step is meant to
change the emitted code.

#### Two tiers, because triviality and self-reference are different questions

The same distinction `TermArg` encodes. A **self-reference** is an expression
question — "is this operand this very phi" — and `block_param_classes` answers it by
class, which is what `commit_terminator_arg_vregs`'s back-edge case already detects.
**Triviality** is a storage question, and the answer splits:

- **Tier 1, unconditional.** Every predecessor passes the *same VReg*. The value is
  already in one register on every path reaching the block, so removing the parameter
  needs no new definition and no cost model. This is where a class emitted once in a
  dominating block lands, which is most of a `pressure` function's parameters.
- **Tier 2, cost-gated.** Predecessors pass *different VRegs of the same class* —
  each computed the same expression into its own register, which is the case the
  parameter exists for. Removal needs the target block to re-emit, so it is the
  dial's decision.

Keeping the tiers separate matters for soundness as much as for quality: tier 1 needs
no justification beyond "same register", so it cannot be the thing that reintroduces
`phi_simplify.rs`'s "reads a register only one path wrote". If tier 2 ever has to be
disabled, tier 1 stands on its own.

#### It subsumes `propagate_block_params`

`egraph::algebraic::propagate_block_params` merges a single-predecessor block's
parameter with its incoming argument, and refuses unless the argument is *constant* —
its own comment says why: "merging with non-constant values can cause extraction to
schedule computations in the wrong block (the source computation may not dominate the
target block)". That is the dominance problem step 2 solves by re-emitting, so step 2
does the same job for **every** argument rather than only constants, and does it by
removing the parameter instead of merging two classes.

Deleting it also removes a hazard class rather than adding one. Merging is what makes
**two parameters of one block share an e-class**, which is why `build_phi_copies`
needs its `params_copied` dedup and why `remove_terminator_arg_operands` had to be
keyed on argument index instead of VReg (`f5ab27a`, a wrong-code bug). No merge, no
shared-class parameters, no dedup.

Check it against the corpus rather than assuming: the merge also feeds constant
folding across inlined boundaries, so confirm the folding still happens once the
parameter is gone instead of merged.

#### `src/compile/phi_simplify.rs` is deleted by this step

269 lines behind `--enable-phi-simplify`, off by default, superseded not fixed. Its
private `terminator_edges` duplicate goes with it. Leaving a second, class-based
trivial-phi pass in the tree next to a working one is how a later session picks the
wrong one.

#### This step needs a code-quality metric, and there is none

Everything before it was judged by correctness and capacity, both of which exist.
Step 2's tier 2 is the project's first genuine **cost** decision: recompute here
versus copy there. `compare_ref.sh` cannot referee that — it reports whether a
program is correct and whether it allocates, not whether the code got better. So
ROADMAP P0-Measurement stops being a "nice to have once the refactor is done" and
becomes a prerequisite of *this* step: instruction count, `.text` bytes,
spill/reload count per program, with checked-in baselines so a diff shows a
regression. Without it the dial's default is unfalsifiable and the 85-94% number
measures parameters removed rather than code improved.

### Step 3 notes — the remaining operands

`Load.addr`, `Store.addr` and `.val`, `Call.args`, and the result classes
(`Load.result`, `Call.results`), the same `TermArgs` treatment step 1 gave block
arguments. Two members of the set are easy to miss because they are not in the
obvious list:

- **`Branch.cond`.** Resolved through the class map in two places nobody counts:
  `barrier::mark_branch_cond_barrier`, to force the flags-producing instruction into
  the last barrier group, and `compile/mod.rs`'s Phase 4b, to find the `Proj1` + ALU
  pair that must sort to the end of its group. Both run before the splitter, so both
  are currently right; both are resolutions all the same.
- **`Ret.val`.** Step 1 left it deliberately. It cannot simply become a VReg: lowering
  reads the *class* to fold a constant return straight into the ABI register
  (`egraph.get_constant`), which exists because trusting the register assignment there
  emitted nothing at all for `return 0` after a call, so `main` returned whatever
  `printf` did. So `Ret` wants the same both-fields shape `TermArg` has, or the
  constant has to be decided at commit time and carried.

Step 3 is what makes `verify_cfg_schedule_agreement` vacuous so it can go, and with
it both of the post-splitter disagreement shapes recorded under prereq A.

### Step 3b notes — block parameters get VRegs

The phi seam has two ends and step 1 fixed one. `cfg::resolve_block_param_vreg` still
answers "which VReg does this parameter live in" from **four** ordered fallbacks, and
three call sites consult it (`build_phi_copies`,
`compute_copy_pairs_from_schedules`, `collect_block_param_vregs_per_block`). Two
passes deriving it separately is what `021d4ed` and `29e796d` were.

Linearization already decides it once, in one place, and records it in
`block_param_vregs` — so this is the same move step 1 made: give `BasicBlock` a
`param_vregs` beside `param_types`, write it where the decision is made, and collapse
the four sources to one.

It is listed after step 3 rather than before because **step 2 does not need it**: the
self-reference test is a class question (see step 2's two tiers), so step 2 can land
without touching this chain. It is listed *before* step 4 because step 4's deletion
list already contains `block_param_vregs` and `block_param_vreg_overrides`, and
deleting them needs something to have replaced them.

One thing to get right, learned from step 1: the splitter truncates a parameter's
segment but never renumbers its `Op::BlockParam` dst, so a committed parameter VReg
stays valid modulo coalesce aliases, which are already applied downstream. The
`param_vregs` list is therefore stable the way `TermArgs::Committed` is — and, like
it, is not the post-split authority for a parameter routed through a slot.

### Step 4 notes — what to delete

Per-block snapshots, the three-times-patched map, `block_param_vregs`,
`block_param_vreg_overrides`, `class_emitted_in`, and the `value_defs` guard in
`split.rs`. Judged by LOC removed with no behaviour change.

It also unblocks the **slot-level verifier**: a reload must produce the class that was
stored. That check cannot be built on the current map, which reports false positives
on the splitter's own immediate store/reload pairs because it has collapsed each class
to one VReg. It is the only check that would see a spill routing a value to the wrong
cell, which nothing downstream can: the displacement is well-formed and the store
writes it.


---

## Soundness: three checks that do not exist yet

The goal is the best x86-64 code, and every quality lever here moves a value between
registers, slots and recomputations — so each step should leave behind the invariant
that says it did so correctly. `BLITZ_VERIFY` covers structure and def-before-use;
these three gaps are what it does not cover, each with the step that should close it.

**None of them adds a gate.** The battery is four runs and stays four runs: new
invariants go *inside* the runs that already happen — 1 as a `debug_assert` the
`checked` profile already carries through every harness, 2 inside
`verify_register_sharing`, which only executes under `BLITZ_VERIFY`. The rule to hold
to is that the gate *set* is fixed and checks are added within it. A battery that
grows every time something is learned stops being run between every change, and
one-change-at-a-time is what makes attribution possible here.

**1. `vreg_types` completeness — cheap, and step 2 makes it urgent.** There is no
assertion anywhere that every VReg a schedule names has a type. A missing entry is
not a pessimism: `lower.rs`'s `result_size` falls back to `OpSize::S64`, which turned
a flags-only 32-bit compare into `cmp r8,rdi` against a zero-extended `mov edi,-2`,
so `14 < -2` was true (`9207141`). The cause was a class **re-emitted in a later
block**, whose VReg the function-wide map does not keep because the per-block restore
is an `insert_single` that replaces segments. Step 2's tier 2 deliberately *increases*
re-emission — that is the remat lever — so it increases traffic through exactly the
mechanism that produced that bug. A one-line invariant, over every schedule after
linearization, would have caught it directly and costs nothing to keep.

**2. `verify_register_sharing` cannot see an illegal coalesce, by construction.** It
canonicalizes every VReg through `coalesce_aliases` before it counts registers
(`verify.rs:992`), so by the time it looks, two values that were merged are one VReg
and there is nothing to compare. Coalescing is where three wrong-code bugs landed
(`021d4ed`, `29e796d`, and the `briggs_admits_illegal_merge.c` investigation), and it
is the pass with the least verification. Closing it means comparing each merge against
liveness measured **on the schedules as emitted**, not on the post-coalesce naming.
Best done with step 6, which is already touching both allocators.

**3. Nothing checks that a value is *right*, only that it is written.** Stated in
`CLAUDE.md` and worth keeping stated: the machine verifier is satisfied by a register
that holds somebody else's value. That is the differential harness's job, which is why
`run_diff.sh`'s two oracles and `gen_c.py`'s UB-freedom are load-bearing rather than
nice to have, and why no green verifier run should be read as "codegen is correct".

---

## Step 0: slot numbering — DONE

The three seams were `pre_spill_slots`, the splitter's per-round `first_slot`, and a
shift in `compile/mod.rs` that told the global allocator's slots from the splitter's
**by number range** (`slot_shift` / `pre_allocated_slots`). That last one was only
safe because `run_phase5` never allocates a slot, so `global_alloc_slots` was always
0 and the shift loop never ran — a hazard that would have fired the moment step 5
gave the allocator a spill loop, silently misclassifying its slots.

`regalloc::slots::SlotAllocator` is now one allocator per function, threaded to all
three passes that spill, and it records a `SlotOwner` per slot. The consequences:

- No range discrimination and no shift: `insert_early_barrier_spills`,
  `plan_splits` and `insert_spills` all call `alloc(owner)`, so the numbers are
  distinct by construction and the frame reserves exactly `slots.count()`.
- `SplitPlan::slots_allocated` and `GlobalRegAllocResult::spill_slots` are gone.
  Neither can report a private numbering, so a spill loop in `run_phase5` has to
  take its slots from the one it is handed.
- `BLITZ_DEBUG=slots` names the owner beside every access, which is what decides
  whether a suspicious access is suspicious: an early-barrier slot is stored and
  reloaded inside one block, a splitter slot spans blocks.
- A `debug_assert` in `compile/mod.rs` rejects a spill op naming a slot no pass
  allocated. Nothing downstream can see that shape on its own — the displacement is
  well-formed and the store writes it, so the machine verifier and the slot dump
  both read it as an ordinary slot.

Judged by byte-identical output rather than a pass count, since it changes no
behaviour: over the 440 lit programs and 180 generated ones at both levels, 982
emitted-asm comparisons against the pre-change compiler are identical, and every
failing pair fails identically (46 overshoots, 3 pre-existing division-projection
assertions). The overshoot message gained the count of slots already committed.

---

## Step 5: the function-scope allocator cannot spill

`run_phase5` in `src/regalloc/global_allocator.rs` is `if converged { Ok } else
{ Err }`. There is no spill loop. Meanwhile `src/regalloc/allocator.rs` — the
per-block path, used only for single-block functions — has one:
`for round in 0..=MAX_SPILL_ROUNDS`, with an `interval_color` fallback. The
capability exists in the path that barely runs and is absent from the primary one.

This inverts the contract. A register allocator's contract is "always succeeds,
worst case with a lot of spill code." Blitz's is "the splitter must be perfect" —
which is why `split.rs` is 3142 lines, why it was measured marching one overshoot
forward for 27 rounds and re-choosing the same victim four times with a fresh slot
each time, and why every remaining corpus failure is a *compile error* rather than
merely slow code.

It also makes every splitter change a correctness gamble instead of a quality knob,
which is the wrong footing for a project whose goal is code quality.

Once this lands, the splitter's termination problem largely dissolves: it iterates
with a round limit and treats an empty plan as convergence, so today it cannot tell
"done" from "out of budget". Behind a real spill loop it becomes a heuristic that
improves code rather than a gate that must succeed, and the round limit stops being
load-bearing.

---

## Step 6: two allocators

`allocator.rs` (per-block, spill rounds, interval colouring) and
`global_allocator.rs` (function-scope, primary) do the same job twice, with the
capabilities split the wrong way. A single-block function is a special case of the
general one, not a separate algorithm. Fold after step 5, or the spill loop gets
written twice.

Beyond merging the two files, the single-block path is the last consumer of two
helpers that duplicate the multi-block path's answers: `cfg::compute_copy_pairs`,
which resolves parameter VRegs through the class map where the multi-block path uses
`compute_copy_pairs_from_schedules`, and `cfg::collect_phi_source_vregs`. One
algorithm means one derivation of the copy set, which is the same "two passes deriving
it separately" hazard as everywhere else in this document.

This is also the natural place for soundness gap 2 above: the merge-versus-liveness
check needs to sit inside whichever allocator survives.

---

## Step 7: `Op` is three enums wearing one hat

89 variants: 36 pure IR, 42 x86 machine, and **11 pseudo/markers that define no
value** — `has_no_result()` exists to say so, consulted in 6 places.

Every 2026-08-03 bug in that area was "a pseudo-op sits in a structure that assumes
an op defines a value": `TerminatorArgs`'s phantom `dst` taking a colour,
`BlockParam`'s marker *position* read as a def point, spill-store dsts taking
registers. Splitting into `PureOp` / `MachOp` / a schedule-level `Pseudo` makes those
unrepresentable instead of guarded — the same kind of defect as the CFG one, where
the type system has stopped helping.

Lowest priority, and a natural follow-on to steps 1-4 since those already change
what the CFG holds.

**One open question underneath it: can `Op::BlockParam` leave the e-graph
altogether?** It is the root-cause statement of
this whole document — a position-dependent value stored as a position-free e-node — and
after steps 1-4 the CFG names both ends of every phi by VReg, so the node looks
vestigial. It is not yet: `EGraph::block_param_classes` answers step 2's
self-reference test, `extract.rs` has a tie-break preferring a `BlockParam` node over
a non-`BlockParam` candidate, and `cost.rs` weights it 0.0. Deleting it needs each of
those re-expressed without it. Worth checking before step 7 rather than assuming
either way; if it can go, `block_param_map`, `block_param_classes` and
`rewrite_block_params` go with it and "one e-class is one expression, not one value"
stops being a hazard the pipeline has to remember.

---

## State when this was written (2026-08-03, `81c0f2e`)

Gates: 924 unit, 440 lit at `BLITZ_VERIFY` off/1/strict, 281 differential + `cc`.
51.9k lines of Rust across 98 files.

Generated corpus, release build, 60 seeds per shape, per (seed, level) pair:

| shape | passing | failures |
| --- | --- | --- |
| `mixed` | 58/60 | 2, both `-O0` |
| `args` | 53/60 | 8, mostly `-O0` |
| `pressure` | 24/60 | 36, **all** `-O1` |

**Every remaining failure on every shape is `register pressure overshoot`.** No
wrong-value program is open and `tests/fuzz/findings/` is empty — a first for this
corpus. Do not widen the generator past 60 seeds per shape until these are fixed.

Redundant block parameters, `tests/fuzz/count_trivial_phis.py`:

| program | parameters | redundant |
| --- | --- | --- |
| `pressure` seed 22 -O1 | 97 | 88 (90%) |
| `pressure` seed 5 -O1 | 197 | 186 (94%) |
| `pressure` seed 7 -O1 | 85 | 79 (92%) |
| `mixed` seed 58 -O1 | 133 | 120 (90%) |
| `regalloc/coalesce_pair_from_schedule.c` -O0 | 247 | 212 (85%) |

On `pressure` seed 22 the loop header carries 28 parameters of which **4** are real,
and block 20 carries 28 of which **none** — a single-predecessor pass-through block
whose incoming edge is the 28-argument terminator that stalls the splitter.

## Cleanup: the per-pass flags are a debug facility wearing product clothes

Seven `--enable-X` / `--disable-X` pairs mirror the seven `enable_*` fields on
`CompileOptions` one-for-one, and the `-O` levels are what actually configure the
pipeline: `o0()` sets every pass off, `o1()` sets every pass on except
`phi_simplify`. So the flags add no configuration the levels do not already express;
what they add is the ability to deviate from a level.

**That ability has earned its keep, for one reason.** Bisecting the pass set is
`CLAUDE.md`'s fifth debugging technique and the cheapest attribution tool in a backend
whose main hazard is wrong code — `-O0 --enable-inlining`,
`--disable-store-forwarding`. Keep it.

**Two things about the current shape are wrong, though.**

- **The reachable configuration space is 2^7; the tested one is 2.** Every gate — 440
  lit, 281 differential, the fuzz corpus at 60 seeds a shape — runs `-O0` and `-O1`
  and nothing else. 15 of the 440 lit tests pin a pass flag, and those are the only
  mixed configurations under test at all.

  **The answer is not to test more of it.** `-O0` and `-O1` are the supported
  configurations; the toggles are a debugging facility, and a pipeline reached only by
  a hand-picked combination of them is not a configuration this compiler claims to
  compile correctly. Gating 128 pipelines would multiply the battery to buy confidence
  in configurations nobody ships, and the battery's length is a real constraint — it is
  run between every single change, which is what makes one-change-at-a-time affordable.

  What follows from that is the opposite of more testing: stop presenting the toggles
  as options. Deviating from a level should look like reaching for a debugger, not like
  choosing a build configuration.
- **It is seven independent dials with no derived default**, which is the inverse of
  this project's own rule (*prefer one dial with a derived default*). The other
  development switches already live in the environment — `BLITZ_DEBUG`,
  `BLITZ_VERIFY`, `BLITZ_DEBUG_FN` — and a single `BLITZ_DISABLE=licm,dse,inlining`
  derived off the `-O` level would keep the *shipped* surface at exactly
  `{O0, O1}` while leaving bisection as easy as it is now, and would name the
  mechanism as what it is.

**`--enable-phi-simplify` is a different animal in the same cage:** a feature gate on
an unfinished pass, off at every level, and the reason `phi_simplify.rs` has sat in
the tree since `02be4ae` being neither used nor deleted. Step 2 removes both.

Also: the six `licm/*.c` tests pass `--enable-licm` although LICM is already on at
`-O1`, which is the level `run_tests.sh` compiles at. Harmless, and a small sign that
the flags read as "how you turn a pass on" rather than "how you deviate from a level".

## Not part of this roadmap

- ~~**A benchmark harness**~~ — **moved in.** It was written here as "worth having and
  not a prerequisite… becomes the binding constraint once steps 1-5 are done", on the
  grounds that every step is judged by correctness plus capacity. That holds for steps
  1, 3, 3b and 4, which change no behaviour or only delete. It does **not** hold for
  step 2: its tier 2 is a cost decision, recompute here versus copy there, and no
  existing gate can referee it. See "This step needs a code-quality metric" under the
  step 2 notes. ROADMAP P0-Measurement is therefore a prerequisite of step 2, not a
  follow-on to step 5.
- **Redo the constant-remat pass.** Unblocked by step 4, and the next quality lever
  after step 2. It failed twice on the seam step 4 removes, never on its policy.
- **Shape A** (`gpr_overshoot=1`, mostly `-O0`, ~8 pairs): real pressure at a call
  with twelve argument operands, where the splitter's overshoot marches forward two
  instructions per round. Step 5 turns it into spill code. Diagnosis in
  `docs/terminator-args-next-steps.md` item 10.
- File sizes. `compile/tests.rs` (3751 lines), `compile/mod.rs` (2043),
  `egraph/algebraic.rs` (2289) are untidy with no evidence of harm, and a rule set
  being long is fine.
