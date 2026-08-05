# Refactor roadmap

The structural work, in the order it should happen and with the reason for that
order. Written to survive across sessions: it assumes no context beyond the repo.

Judge every step on the full battery (`cargo test --all-targets --workspace`,
`bash tests/lit/run_tests.sh` at `BLITZ_VERIFY` off/1/strict,
`bash tests/lit/run_diff.sh`) and per **(seed, level) pair** with
`tests/fuzz/compare_ref.sh <ref> 60 <shape>`, never on a bare pass count.

## Cheap things that make the rest easier

None of these changes behaviour. Each one shrinks the surface or the risk of the
steps below, and they are worth doing first rather than carrying through a rewrite.
A, B and C were done before step 1; **D was a prerequisite of step 2** and is done.

**A. A CFG-versus-schedule agreement check, under `BLITZ_VERIFY`.** DONE, and
**deleted in step 4** once the fallback it guarded was measured dead. What
follows is what it found while it existed.
`verify::verify_cfg_schedule_agreement` compares, at every position both
representations name, the VReg a class resolves to through the map against the VReg
the schedule carries as an operand: an effectful op's role operands against the
barrier consuming it. Enforced at construction, where it is clean over the whole
suite; reported after the splitter, where it is not. It was expected to go with
step 3; it does not -- see the step 3 notes for what actually retires it.

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

Step 3 made the construction-time half agree by construction as well, and step 4
retired the check outright: the fallback it guarded was measured to fire **0
times** over the whole lit corpus and 30 generated programs at both levels, so
`lower_effectful_op` no longer has one. The barrier is the only answer, and an
absent role operand is a `CompileError` naming the op instead of a quiet lookup
that may name a stale register.

Also found and left alone at the time, since it was a behaviour change and that
step was not: `populate_effectful_operands` resolved at
`ProgramPoint::barrier_point`, the barrier's index **plus one**, while
`lower_effectful_op` resolves at the raw index. Step 3 removed the question:
`populate_effectful_operands` resolves nothing at all now.

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

**D. One derivation of the CFG's edges. DONE, and a prerequisite of step 2.**
`cfg::successor_ids` answers it for one terminator, `cfg::successor_indices` for the
whole function, and `cfg::predecessor_indices` inverts that — all three over
`barrier::terminator_edges`, which was already the general form: each successor with
the arguments that edge carries.

Eight derivations went, not the five the scan found. `cfg::compute_rpo`'s inline
closure and `regalloc::global_liveness::cfg_successors` were *character-identical*
over 25 lines; `cfg::compute_idom`, `licm::build_predecessor_map` and
`licm::detect_back_edges` each had the same walk written differently;
`dce::block_successors` returned `BlockId`s rather than indices;
`cfg::compute_loop_depths` and `algebraic::propagate_block_params` had one apiece
that the window scan missed because neither is named for what it does.
`phi_simplify`'s own `terminator_edges` went with them.

This was a prerequisite rather than a cleanup because **step 2's core question is
per-predecessor**: "does every predecessor pass the same VReg at this position". Left
alone, step 2 would have reached for `build_predecessor_map` and become the *sixth*
consumer of a fact five passes already derived separately — the defect the rest of
this document exists to remove, reintroduced by the step meant to benefit from
removing it.

Same shape as A, B and C: behaviour-neutral, and byte-identical over the lit corpus
at both levels (`tests/run_identity.sh`, 671 identical of 674).

Not worth doing first: splitting the large files (no evidence of harm), and the `Op`
split -- that is step 7, and doing it early fights steps 1-4, which change what the
CFG holds.

## Duplication: do it first, except where first makes it worse

Found by a 10-line-window scan over `src/` and `crates/`, filtered to production code
and then read. **Cleanups go before new work** — that is why A, B and C preceded step 1
and why step 0 preceded it too, and it is what keeps a rewrite from carrying somebody
else's mess through it. Every item here is behaviour-neutral, so each is one commit
gated by `tests/run_identity.sh` and the battery, which is the cheapest kind of change
this repo can make.

**The exception, and it is mechanical rather than a matter of taste: dedupe now only
where the result is what the owning step would have produced anyway.** Where deduping
first means inventing a shared helper that the step then deletes, you have not removed
a duplicate — you have added a third thing to keep in sync until the step lands, and
paid a rewrite for the privilege.

| what | when | why |
| --- | --- | --- |
| Five derivations of the CFG's edges | **DONE** (prereq D) | step 2's core question is per-predecessor; left alone it becomes the sixth consumer |
| `apply_splits_for_overshoot`'s 25 parameters | **DONE** | steps 2 and 5 both add state to the splitter, and each new piece threads through 25-argument signatures |
| `verify.rs`'s private `chase_alias` | **DONE** | one line, and a verifier's own copy of what it verifies is the worst place for a drift |
| `block_id_to_idx` rebuilt in six files | **DONE** | trivial, and steps 3b/4 churn that area |
| Two pairs of near-identical rule bodies | **DONE** | `egraph/` only, no interaction with any step |
| The generic-IR-op set in `lower.rs` and `cost.rs` | **wait for step 7** | the list *is* `PureOp`; a shared constant now is a third copy the split deletes |
| `allocator.rs:319` / `global_allocator.rs:531` | **wait for step 6** | a shared helper now is deleted by the fold |

Five of the seven landed before step 2, in five commits, each byte-identical over the
lit corpus at both levels. The two that wait are written up under the steps that own
them.

The three items not covered as prereq D, and what each turned out to be:

**`split::apply_splits_for_overshoot` took 25 parameters at 3 call sites, with 21
identical at every one.** Now four: a `RoundCtx` of what is fixed for the whole
`plan_splits` call, a `BlockCtx` of what the current block's liveness scan produced, a
`PlanAccum` of everything a split writes to, and the `Overshoot` itself — index,
class, excess and the `SplitScope` the finding path wants. The accumulator is what
steps 2 and 5 add their state to; before, each new field meant a 26th parameter at
three sites.

**`block_id_to_idx` was rebuilt inline in six files, fourteen sites**, always the same
map. `cfg::block_id_to_idx` is the one builder. Minor, and the sort of minor that
shows up in a profile: O(blocks) each time, and `commit_terminator_arg_vregs` built a
second one while `compile/mod.rs` already held it.

**Two pairs of near-identical rule bodies.** `algebraic.rs`'s
`apply_sub_zero_eq_ne_rules` / `apply_add_const_zero_eq_ne_rules` shared 36 lines and
are now one `apply_icmp_zero_rules` taking the inner operation and a closure from its
two children to the pair the new comparison names; the Sub rule is a one-liner and the
Add rule is its constant negation. `known_bits.rs`'s `Shl`, `Shr` and `Sar`
by-constant arms shared the same 25-line preamble, now `shift_by_constant`, which is
also the one place the out-of-range amount is rejected.

### Do not merge the two local liveness passes

`regalloc::liveness::compute_liveness` and `split::compute_local_liveness` are both
backward scans over a schedule from a live-out seed, and they look like an obvious
dedup. **They must differ.** The splitter's version counts a block's parameters as
live over the `BlockParam` marker run; the allocator's deliberately does not, because
extending the ranges the allocator *spills against* forces real spill code — measured
at 36 → 33 on the corpus, with a new pressure failure and an `-O0`/`-O1` behaviour
divergence, and reverted (`b5c0667`'s note). Modelling it in the splitter's
measurement and the interference graph only is what worked. A dedup pass that unified
them would silently reintroduce that regression, and the gates would show it as a
capacity loss with no obvious cause.

---

## Order, and why

| # | step | size | why here |
| --- | --- | --- | --- |
| 0 | Slot numbering gets a real representation | small | **DONE** — was a documented landmine, and a prerequisite for step 5 |
| 1 | Terminator args become VRegs in the CFG | medium | **DONE** — the representation defect, smallest slice with the biggest payoff |
| — | Prereq D and four cleanups | small | **DONE** — behaviour-neutral, before step 2; see "Duplication: do it first, except where first makes it worse" |
| 2 | Trivial-phi elimination over those VRegs | small-medium | **TIER 1 DONE** — measured: fewer spills and reloads everywhere, capacity unchanged. Tier 2 open |
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

## Steps 1-4: the CFG should state which register holds each value

**Not "the CFG should hold VRegs, not ClassIds", which is what this heading said
and which the design settled against in step 1.** A committed operand carries
both: the `ClassId` because expression identity is what `build_phi_copies` asks
to tell an argument that *is* its target parameter from one that merely equals
it, what `Ret` lowering asks to fold a constant return, and what DCE and
canonicalization walk; the VReg because none of those answer "which register,
here". What steps 1-4 remove is the *resolution machinery* between the two, not
the class. Whether the class itself can leave is step 7's question.

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

### Step 2 notes — trivial-phi elimination — TIER 1 LANDED

`src/compile/phi_removal.rs`, on by default. What shipped and what did not:

- **Both tiers are in.** Tier 1: the source class's emitter dominates the block,
  so nothing is re-emitted and only the copies go. Tier 2: it does not, the block
  recomputes, and the dial decides -- removal when the class's extraction cost is
  at most one copy per incoming edge plus the parameter's slot in the clique.
  Tier 2 is tried first and tier 1 is the fallback, since the acceptance test
  below is a property of the whole removal.
- **The removal is not local, and the pass now proves its own result.** Removing
  a parameter means unioning its class with the source's -- the block's own uses
  still name the parameter, and a class whose only node was that parameter has
  nothing left to lower. But a union changes what *every other* list naming
  either class means, and composing several put a `CallResult` where a block
  re-emitted it: a second barrier instruction for one effectful op, which
  lowering asserts against. Predicting that composition was the wrong shape of
  fix. The pass removes, re-extracts, linearizes, and then checks that no block
  emits more barrier results than its own effectful ops produce; if it does, the
  CFG and e-graph go back exactly as they were. `Function` derives `Clone` for
  that reason.
- **Measured** (`tests/run_codesize.sh`, totals against the pre-step baseline):
  `lit` -1.6% instructions / -3.9% reloads, `fuzz` -3.1% instructions / -8.3%
  spills / -6.3% reloads, `bench` +1.1% instructions but -13% spills / -8.5%
  reloads. Per program on `bench` it is genuinely mixed: `hash_table` -O1 is
  -9.2% instructions and -29% spills, `binary_search` -O1 -42% spills, while
  `sort_insert` is +11% and `queens` +9%. Removing a parameter replaces a copy
  per edge with a live range that spans to the use, and on a recursive function
  that range now crosses a call.
- **Capacity is unchanged, with both tiers**: 180 (seed, level) pairs on all
  three shapes, 0 regressed and 0 fixed. The step's claim that this clears 36 of
  the 46 capacity failures does not survive measurement.

  **Why, counted on pressure seed 22** (97 parameters, which
  `count_trivial_phis.py` calls 90% redundant): 57 have predecessors passing
  *different classes* -- real phis, nothing can stand in for them -- and 27 carry
  a placed value (a `LoadResult`, `CallResult`, `Param` or `BlockParam`), which
  has no re-emission available. 13 were removed. The dominance and cost gates
  rejected nothing at all, so neither tier's condition is what binds. The 85-94%
  figure is measured on the pre-extraction IR by a tool that asks a weaker
  question than soundness does; at the point where the removal is sound, most of
  those parameters are not trivial.

  **The one lead worth following**, measured and then reverted: vetoing placed
  values even where the emitter *dominates* is stricter than soundness needs, and
  lifting that took seed 22 from 13 removals to 40. It also stopped pressure seed
  19 compiling at -O0 -- the parameter was holding apart a live range the
  allocator could then not colour. That is a splitter question, so it belongs
  after step 5 rather than here.
- `propagate_block_params` is still in place. It was to be subsumed, and tier 1
  does not subsume it: it merges a single-predecessor block's parameter with a
  *constant* argument, which is a class-level merge this pass does not make.

### Step 2 notes — the original design

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

#### This step needs a code-quality metric — DONE, and it moved the picture

Everything before it was judged by correctness and capacity, both of which exist.
Step 2's tier 2 is the project's first genuine **cost** decision: recompute here
versus copy there. `compare_ref.sh` cannot referee that — it reports whether a
program is correct and whether it allocates, not whether the code got better. So
ROADMAP P0-Measurement was a prerequisite of *this* step, and it is now
`bash tests/run_codesize.sh`: instructions, `.text` bytes, spills and reloads per
(program, level), 888 baseline rows across `lit`, `bench` and `fuzz`. Without it
the dial's default is unfalsifiable and the 85-94% number measures parameters
removed rather than code improved.

**The first table said something this step has to account for.** On the kernel
corpus `-O1` emits *worse* code than `-O0` on 7 of 15 programs, spills and
reloads every time — `hash_table` 223 instructions at `-O0` against 415 at `-O1`,
`matmul` 0 reloads against 23. Attributed with `FLAGS=--disable-licm`: LICM is
60% of the reloads and inlining most of the rest, with the other four passes flat
(ROADMAP P1 carries the per-kernel numbers).

That is a *policy* gap rather than a representation one, so it does not reorder
this document — but it bounds the claim at the top of this step. Removing 85-94%
of block parameters removes the phi copies and the terminator-argument clique; it
does not shorten a live range that spans a loop because LICM put the value there.
Expect step 2 to move the parameter counts and the `fuzz` capacity failures, and
expect the `bench` reload counts to stay put until the hoist decision consults
the same pressure the splitter measures. Judge it on the rows it should move.

### Step 3 notes — the remaining operands -- DONE

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

Step 3 was expected to make `verify_cfg_schedule_agreement` vacuous so it could
go. **It does not, and the prediction was wrong about which change retires it.**
The check compares the class map's answer against the barrier's operand; slices
1-3 make the two agree *at construction* by seeding the barrier from the CFG, but
after the splitter the barrier's operand is rewritten while the map is patched
separately, and that difference still matters because `lower_effectful_op` falls
back to the map wherever a role operand has no register. The check guards that
fallback. It goes when the fallback does, which is step 4 deleting the map.

**Do it in this order, one commit each, gates between.** The fields are
independent, and the cheap ones first buy the shape before the hard one needs it:

1. `Load.addr`, `Store.addr`, `Store.val` -- one carrier each. **DONE.**
2. `Call.args` -- a list of the same carrier. **DONE.**
3. `Load.result`, `Call.results` -- *defs*, not uses. **DONE.**
4. `Branch.cond` -- one field, wide to land. **DONE.**
5. `Ret.val` -- last. **DONE.**

**Slice 1, as it landed.** `EffOperand` is the carrier: `Class(ClassId)` while
the IR is still being transformed, `Committed { class, vreg }` from
`cfg::commit_effectful_operand_vregs` onward, which runs beside
`commit_terminator_arg_vregs` at the end of linearization and resolves through
the *block's own* snapshot for the same reason that one does. Every segment
linearization makes is full-range, so the point inside the block cannot
distinguish two answers -- the block is what selects. `populate_effectful_operands`
now reads the committed VReg for a `Load`'s address and a `Store`'s operands
instead of resolving the class at the barrier point; `Call` still resolves, which
is slice 2. An operand with no VReg is a compile error naming the op rather than
a dropped operand, since the barrier's roles are positional and a dropped one
takes a `Store`'s value for its address.

Two things followed. `phi_removal` uncommits the operands along with the
terminator arguments, because a removal hands the CFG back to linearization and a
VReg from the previous one is an answer to a question the next one re-asks. And
`verify_cfg_schedule_agreement` keeps only its post-splitter half for these
roles: before the splitter the barrier is now a copy of what the CFG states, so
the two sides agree by construction.

Judged as predicted: **768 identity comparisons byte-identical** (708 lit, 60
generated at both levels), 0 differing and 0 differing in status;
`run_codesize.sh --check` 888 rows unchanged. 934 unit, 474 lit at `BLITZ_VERIFY`
off/1/strict, 298 differential + `cc`.

**Slices 2 and 3, as they landed.** `Call.args`, `Load.result` and
`Call.results` take the same carrier, and the commit function is
`cfg::commit_effectful_vregs` -- results included, because the VReg a `Load`
writes is the `dst` of the barrier instruction linearization emitted for it, so
committing it lets a consumer find that instruction without asking the map which
VReg carries the result.

What that deleted is the point: `populate_effectful_operands` no longer takes the
e-graph, the class map or a block index, and `add_call_precolors_for_block` no
longer takes any of the three either. Neither pass resolves a class any more.

**Slice 2 changed emitted code, and the disagreement it found was already
there.** `add_call_precolors_for_block` pinned each argument to its ABI register
through the *function-wide* map at the block's entry, while the barrier resolved
the same argument through the block's own snapshot at the barrier point -- and
for a class re-emitted per block those answer differently, so the pin landed on a
VReg the block does not define. It is the same stale answer `build_barrier_maps`
had, and `regalloc/call_arg_reemitted_in_block.c`, the regression test written for
that one, is one of the four lit programs that move. Four lit programs and one
generated program emit different code, all at `-O0`, all a different choice of
which block gets the ABI register directly and which keeps a `mov`. Nothing
regressed: `hash_table.c` -O0 224 -> 223 insts, `mixed-seed28` -O1 6199 -> 6196,
`pressure-seed1` -O0 6524 -> 6523, 885 other codesize rows unchanged, and 90
generated (seed, level) pairs with 0 regressed, 0 fixed, 0 changed kind.

Slice 3 is byte-identical to slice 2: 768 identity comparisons, 0 differing.

**Slice 5 needed no decision after all.** This step's open question was whether
`Ret.val` could become a VReg at all, since lowering reads the *class* to fold a
constant return straight into the ABI register -- so it wanted "the both-fields
shape `TermArg` has, or the constant decided at commit time and carried". The
carrier slice 1 introduced already keeps the class beside the VReg, so
`egraph.get_constant` still has what it asks for and nothing had to be decided
early. `Ret.val` is committed by `commit_terminator_arg_vregs` rather than by
`commit_effectful_vregs`, because its resolution needs the parameter overrides
that function already holds -- a `Ret` of this block's own parameter whose VReg
linearization minted fresh is not in the snapshot, which predates it.

`append_terminator_args` lost its block index, the e-graph, the class map and the
override map with it. **No point lookup survives in `compile/barrier.rs` or
`compile/mod.rs`**: neither file imports `ProgramPoint` any more. What
`build_barrier_maps` still takes the class map for is a different question --
which VRegs of a class this block defines, deliberately point-free.

**Slice 4 replaced the two derivations of the branch condition with one**, and
they turned out to agree everywhere measured. `mark_branch_cond_barrier` asked
the block's own snapshot at the block's *entry*; Phase 4b asked the
*function-wide* map at the block's *exit*. A flags-typed class is re-emitted in
every block that names it -- linearization forces that, since EFLAGS cannot cross
a block boundary -- so the function-wide map holds whichever block emitted it
first, and the two could differ. On this corpus they do not: 768 identity
comparisons, 0 differing. Both functions lost their `block_idx`, and
`mark_branch_cond_barrier` lost the e-graph and the map as well.

**State the prediction before each one.** These resolutions all run before the
splitter, where they are correct today, so each slice should be byte-identical on
`tests/run_identity.sh` and clean on `tests/run_codesize.sh --check`. A slice that
changes output has found a disagreement that was already there -- which is the
point of the step, but it means stopping to explain the diff rather than
recording a new baseline. Step 2's estimate went unchecked for a year because
nothing could check it; every step from here says what it expects first.

### Step 3b notes — block parameters get VRegs — DONE

The phi seam has two ends and step 1 fixed one. `BasicBlock::param_vregs` sits beside
`param_types`, written by `cfg::commit_block_param_vregs` at the end of linearization
with the other commits and cleared by `phi_removal` with the rest, since a removal
changes the positions themselves.

**The four ordered fallbacks are three, and which ones survive was measured rather
than argued.** A `BLITZ_DEBUG=paramsrc` probe over the regalloc and bench corpora and
30 generated programs at both levels, 123595 resolutions:

| source | answers | note |
| --- | --- | --- |
| the target block's own `Op::BlockParam` | 122223 | the only one that survives coalescing |
| the override linearization minted | **0** | deleted |
| the class map at the target's entry | 442 | of the 1372 the schedule leaves |
| what linearization recorded | 1370 | now `BasicBlock::param_vregs` |

The override answered nothing because linearization gives exactly those parameters a
`BlockParam` instruction of their own, so the schedule answers first every time. That
is a fourth measurement of the same fact: `ba3f1be` found the parameter overrides
inert in `commit_terminator_arg_vregs` and in `append_terminator_args`, and this is
the third consumer to lose them. What still reads them is Phase 7's snapshot
patching.

**Why it cannot collapse to one source, which the plan above assumed it could.** Split
by call site, `compute_copy_pairs_from_schedules` (post-split, pre-coalesce) has the
schedule and the CFG's record agreeing on **all 61592** resolutions where both
answer, so the record is exact through the splitter — as this section predicted. But
`build_phi_copies` (Phase 7, post-coalesce) has them differing on **49646 of 60631**,
because coalescing renames and the CFG's record is the pre-merge VReg. So the
schedule stays the first source and the CFG's record is the fallback, exactly as
`TermArgs::Committed` is not the post-split authority for an argument routed through
a slot. The class map keeps its place between them for the 442, where a reload
covering block entry is what the block reads; it differs from the record in 2 of
them.

Byte-identical: 768 identity comparisons against step 3, 0 differing; 888 codesize
rows unchanged.

### Step 4 notes — what to delete

Per-block snapshots, the three-times-patched map, `Linearized::block_param_vregs`
(now a copy into `BasicBlock::param_vregs` and read nowhere else),
`block_param_vreg_overrides`, `class_emitted_in`, and the `value_defs` guard in
`split.rs`. Judged by LOC removed with no behaviour change.

**Progress, and two items that are not deletable.**

- **`block_param_vreg_overrides` -- DONE**, see below.
- **The class map's readers in effectful lowering -- DONE.** Instrumenting every
  `.or_else(|| get_reg(...))` in `lower_effectful_op` showed **0 hits** over the
  whole lit corpus and 30 generated programs at both levels: after steps 1-3 the
  barrier's role operands answer every time. All of them are gone, the function no
  longer takes a `ClassVRegMap` at all, and `verify_cfg_schedule_agreement` went
  with them -- it existed to compare the map's answer against the barrier's, and
  there is no longer a consumer of the first. Net -256 lines, byte-identical.
- **`class_emitted_in` is NOT deletable.** It has a live reader:
  `phi_removal::source_dominates`, which is tier 1's dominance test. Deleting it
  means answering "does this class's emitter dominate this block" some other way,
  which is a design change rather than a deletion.
- **The `value_defs` guard in `split.rs` is NOT dead either.** Its comment
  describes a live miscompile it prevents: a parameter passing a dominating
  definition through shares the value's VReg, so routing it to a slot moves the
  *value*, and the predecessor's own initialising store becomes a reload of the
  slot it was about to write. What removes the guard is giving every parameter its
  own VReg, which is a design change, not a deletion.
- **The three-times-patched per-block map -- DONE.** `lower_terminator`'s
  `get_reg` was the last lowering reader, at **463 hits**, all of them the
  single-block path where `append_terminator_args` never runs so the schedule
  names nothing. All 463 name the register `Ret.val`'s *committed* VReg names, so
  `ret_value_reg` reads that instead and the map goes. With it went the whole
  `block_class_to_vreg` construction in Phase 7: the snapshot clone, the
  block-parameter patch and the coalesce-alias rename, 56 lines whose comments
  described three bugs each fix had to avoid. **Phase 7 no longer resolves a class
  to a VReg at all.**
- **`AppliedSplits` and the snapshot replay -- DONE.** `apply_plan_to` accumulated
  every segment and truncation it committed so they could be replayed onto each
  per-block snapshot, and the comment said exactly why: Phase 7 resolved
  effectful-op operands through those snapshots, so without the replay a spilled
  address resolved to its pre-spill register. Phase 7 stopped doing that one
  commit earlier, and every other snapshot read happens *before* the splitter --
  so the replay was mutating data nothing subsequently read. The struct, its
  `replay_onto`, both accumulators and the loop are gone, and the snapshots are
  now read-only pre-splitter data.

**What is left of the class map, and why none of it is dead.** Five readers, each
measured:

| reader | what it answers | why it stays |
| --- | --- | --- |
| `split::plan_splits` / `apply_plan_to` | the segments themselves | the splitter owns the map |
| `assign_param_vregs_from_map` | the function's own parameters at entry | no CFG field states these |
| `collect_block_param_vregs_per_block` | which VRegs are parameters, post-split | contributes **68** VRegs `BasicBlock::param_vregs` does not, all of them splitter reloads covering block entry |
| `resolve_block_param_vreg`, source 2 | a parameter's VReg where the schedule has none | answers **442** of 1372, and differs from the CFG field in 2 |
| the per-block snapshots | which VRegs of a class a block defines | `build_barrier_maps` wants *all* of them, which is point-free by design |

So **the function-wide map cannot be deleted**, and step 4's list was wrong to
say it could -- for the same reason step 3b's "collapse to one source" was wrong.
The splitter's segments are per-point facts about reloads, and no field on a CFG
node expresses them. What step 4 could remove was every *reconstruction* built on
top of the map, which is now gone.

**`block_param_vreg_overrides` is gone.** DONE, as step 4's first slice. It had five
consumers left after step 3b, and a probe over the whole lit corpus at both levels
(708 compilations, 11716 trace lines proving the channel live) found that **the map
is never populated at all** -- linearization's fresh-VReg branch does not fire on
anything we test, and zero of the five consumers ever changed an answer.

Deleting it needed more than that measurement, because "unreachable on the corpus" is
not "unreachable by construction". The construction argument is what made it safe:
the branch that mints a fresh VReg also pushes an `Op::BlockParam` instruction for it
*and* records it in `block_param_vregs`, so the fresh VReg reaches every consumer
through the schedule and through `BasicBlock::param_vregs` regardless. The map was a
third copy of a fact those two already carry. Each consumer now reads the block:
`commit_terminator_arg_vregs` (which is why `commit_block_param_vregs` runs first),
the slot-store filter, the allocator's per-block parameter sets, and Phase 7's
snapshot patch -- the last restricted to parameters whose stated VReg is not what the
snapshot answers with, since `insert_single` replaces a class's segments and patching
one the snapshot already resolves correctly would discard what the splitter recorded.

Byte-identical: 768 identity comparisons, 0 differing; 888 codesize rows unchanged.

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

### What landed, and the prediction it refuted

`allocate_global` is a spill loop: build liveness, color, and where the coloring
does not fit, spill the VRegs it could not place and do it again.
`spill::insert_spills_global` is the function-wide form of the per-block
`insert_spills` -- the difference that matters is that a slot is allocated **once
per spilled VReg** rather than once per block, since a value's def and its uses
can be in different blocks.

**Stated before starting: programs that already allocate stay byte-identical, and
`pressure` goes from 14/30 to at least 25/30. The first held exactly. The second
was wrong, and how it was wrong is the useful part.**

Over the 90-program generated corpus at both levels: **161 identical, 0 differing,
and exactly one status change** -- `args` seed 3 at `-O0`, which had never
compiled, now does (395 instructions in `main`, 63 spills). Everything else that
compiled before compiles to the same bytes, which is the property the loop was
designed to have: its body only runs where the old code returned `Err`.

**`pressure` did not move at all, and a spill loop cannot move it.** Traced on
seed 15, the overshoot *grows* with each round: 3, 9, 12, 15, 18, 21, 23, then
flat. Spilling a value whose live range exists because it is an operand of the
pressure instruction cannot help -- the reload lands immediately before that
instruction and is live there too. This is shape B from
`docs/terminator-args-next-steps.md`: a terminator passing 28 block arguments,
where the arguments *are* what is live at the point that overflows. No allocator
can place 28 values in 14 registers at one instruction; the arguments have to
leave the register file, which is `SplitPlan::operand_removals` and the splitter's
slot routing, not spilling.

So the loop stops as soon as a round fails to reduce the overshoot, and says
which of the three reasons it stopped for. Running the limit out committed 207
slots on seed 15 to buy nothing.

**What this corrects in the section above:** "every remaining corpus failure is a
compile error rather than merely slow code" is true, but the implication that a
spill loop turns all of them into slow code is not. It turns *shape A* into slow
code, which is what `args` seed 3 was. Shape B needs the terminator clique broken.

### Removing block parameters does not break the clique, and this is why step 2 bought nothing

The obvious next lever after step 5 was the placed-value veto in `phi_removal`,
which the handoff had flagged to revisit "after step 5": it refuses to remove a
parameter whose source is a load result, a call result or a function parameter,
*even where the source's emitter dominates the block* and so nothing would be
re-emitted. Lifting the dominating case had been measured once before, taking
seed 22's removals from 13 to 40 and costing `pressure` seed 19 at -O0, which was
read at the time as the allocator being unable to spill.

**Retried with the spill loop in place, and it is still a loss**: `args` seed 15
at -O0 stops compiling and nothing is gained, on exactly the shape the spill loop
cannot answer. Reverted, with the measurement in the code so it is not tried a
third time.

The mechanism is the part worth keeping. **Removing a block parameter does not
shrink the live set at the edge.** It converts a value that was copied into a
parameter into one the block names directly, and that value is live across the
same edge either way -- the parameter was never the thing occupying a register,
the value was. Which answers a question this document has carried since step 2:
why removing 85-94% of a program's block parameters moved capacity by nothing.

So the lever is not fewer parameters. It is fewer *values live at the edge*,
which means taking some of them out of the register file: `SplitPlan::operand_removals`
and `slot_spilled_params`, the splitter's slot routing. What limits that today is
recorded in `docs/terminator-args-next-steps.md`: `detect_blockparam_slot_routing`
finds a parameter through `find_block_param_vreg`, which needs an `Op::BlockParam`
marker in the e-graph, and on `pressure` seed 22's 28-argument edge 16 of the 28
parameters have none -- linearization skips the marker for a parameter whose class
is already emitted when the block has at most one predecessor. **Step 3b changed
what is available here**: `BasicBlock::param_vregs` records a VReg for *every*
parameter, marker or not.

**Tried, and it is not the blocker.** With `find_block_param_vreg` falling back to
the CFG's `param_vregs`, `pressure` seed 15 goes from **0 of its 92 parameters
findable to all 92** -- and emits byte-identical code, on the whole corpus.
The routing decision rejects every one of them, because it routes a parameter
only where a block's parameters of one class outnumber that class's budget, and
no block on this program has that many. Reverted; nothing speculative kept.

**Which corrects the recorded diagnosis for this program.** The old note in
`docs/terminator-args-next-steps.md` describes shape B as a 28-parameter block
clique that slot routing cannot see. On seed 15 the parameters are not the
clique: 50 of the 92 have no VReg in the class map at block entry at all, and the
other 42 share an ordinary value's VReg -- the pass-through case, where routing
the VReg would move the *value* and the `value_defs` guard correctly refuses. So
whatever is live at seed 15's pressure point, it is not a block's parameter list,
and the next step is to name that instruction rather than assume it. The spill
loop's own message ("one instruction whose own operands are what is live there")
is the place to start: make it print the instruction.

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

It also ends a duplication the type system is currently unable to prevent: the set of
generic IR ops that must have been lowered before extraction is enumerated twice, 24
variants each, in `lower.rs` (which rejects them) and `egraph/cost.rs` (which prices
them as unlowerable). That list *is* `PureOp`. After the split both sites match on the
type, and a new pure op cannot be added to one list and forgotten in the other — which
is the same failure mode as `has_no_result()` being consulted in six places instead of
being a property of the type.

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

## State at 2026-08-05 (`6e315ac`), after steps 0, 1 and 2

Gates: 934 unit, 474 lit at `BLITZ_VERIFY` off/1/strict, 298 differential + `cc`.
Code quality has a baseline too: `bash tests/run_codesize.sh --check`, 888 rows.

Generated corpus, per (seed, level) pair at 30 seeds per shape: `mixed` 29/30,
`args` 29/30, `pressure` 14/30. Every failure is capacity, and **step 2 moved
none of them** -- see its notes for the count of why.

The 60-seed figures this section quoted before steps 1 and 2, for comparison:

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
