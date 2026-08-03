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
barrier consuming it, and a terminator's arguments against `Op::TerminatorArgs`.
Enforced at construction, where it is clean over the whole suite; reported after the
splitter, where it is not. Step 1 makes it vacuous, at which point it goes.

**The number, over the 440-test lit suite: 0 disagreements at construction, 16
programs disagreeing after the splitter.** Two shapes, and neither is a live bug
because neither consumer trusts the map first -- an effectful op reads the barrier's
own operand and falls back to the map only where no operand exists at that index, and
a terminator argument is read off `Op::TerminatorArgs` with no fallback at all.

- **Two `SpillLoad`s of one class at the same point**, each with a segment of its own
  and both covering it. `ClassVRegMap::lookup` has nothing to choose by and asserts;
  `lookup_ambiguity` exposes that as a value so the check reports it. Also a missed
  CSE: one reload would do.
- **A reload whose segment does not reach the barrier that consumes it**, so the map
  still answers with the pre-spill VReg -- `v864 = SpillLoad(20)` at index 48 is what
  the `StoreBarrier` at index 152 reads, and the map at 152 says `v30 = StackAddr(0)`.
  Segments are keyed on raw instruction indices, and a later split round inserting an
  instruction ahead of one moves the point it was measured against.

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
| 0 | Slot numbering gets a real representation | small | a documented landmine now, and a prerequisite for step 5 |
| 1 | Terminator args become VRegs in the CFG | medium | the representation defect, smallest slice with the biggest payoff |
| 2 | Trivial-phi elimination over those VRegs | small | 85-94% of every function's parameters; removes 36 of the 46 capacity failures |
| 3 | The remaining `EffectfulOp` operands become VRegs | medium | finishes step 1 |
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

Then trivial-phi elimination is a small pass, sound by construction because phi
operands are *defs* rather than expressions. The question that blocks it today
becomes an explicit cost-model choice: where a removed parameter's value is not
available on some path, the target block rematerializes it — cheap for a constant
or an address, and `src/egraph/cost.rs` already exists to decide. That is the same
mechanism the reverted remat pass needed.

### What it buys, in order of size

1. **85-94% of block parameters go**, with their copies, their cliques, their slot
   routing, and their per-iteration store/reload traffic.
2. **Rematerialization becomes expressible** — the second big lever for
   pressure-bound code, and the reason two attempts at it failed.
3. The splitter's operand rewriting reaches everything, so pressure decisions stick.
4. The bug class that has consumed most sessions disappears, so optimization work
   stops being interrupted by miscompiles.

### Step 1 notes — terminator args

`EffectfulOp::Jump.args`, `Branch.true_args`, `Branch.false_args`.

- The choice is already computed. `compile/mod.rs` linearization builds
  `class_to_vreg` plus `block_class_to_vreg_snapshot[block]`, and
  `barrier::append_terminator_args` already resolves every argument to a VReg and
  puts it in `Op::TerminatorArgs`. Step 1 is to make the **CFG** hold that answer
  rather than have Phase 7 ask again.
- Ordering constraint: the rewrite must happen after extraction and after DCE2
  (which needs `func` mutable and rebuilds index-keyed structures), and before the
  splitter — the splitter's whole value is that its operand rewriting reaches the
  operands.
- `EGraph::rewrite_block_params(keep)` is already written and is the reusable half
  of removing parameter positions: it renumbers `BlockParam` nodes in one pass over
  a drained memo, because the new index of a surviving parameter is usually the old
  index of a removed one and any incremental rewrite collides.
- Judged by the 36 shape-B capacity failures (`pressure` at -O1).

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

### Step 4 notes — what to delete

Per-block snapshots, the three-times-patched map, `block_param_vregs`,
`block_param_vreg_overrides`, `class_emitted_in`, and the `value_defs` guard in
`split.rs`. Judged by LOC removed with no behaviour change.

---

## Step 0: slot numbering

Three seams today: `pre_spill_slots`, the splitter's per-round `first_slot`, and a
shift in `compile/mod.rs` that distinguishes the global allocator's slots from the
splitter's **by number range** (`slot_shift` / `pre_allocated_slots`).

`DEBUGGING-NOTES.md` records why that is a landmine: *"That last one is only safe
because `run_phase5` never allocates a slot today."* So it is both a live hazard and
a hard prerequisite for step 5 — the moment the allocator can spill, it allocates
slots, and the number-range discrimination silently misclassifies them.

Give a slot an owner in its representation rather than inferring one from its index.

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

## Not part of this roadmap

- **A benchmark harness** (ROADMAP P0-Measurement, both items unchecked): instruction
  count, `.text` bytes, spill/reload count, hyperfine, checked-in baselines. Worth
  having and not a prerequisite — every step above is judged by the correctness gates
  plus the capacity numbers, both of which already exist. It becomes the binding
  constraint once steps 1-5 are done and the work turns to pure quality tuning.
- **Redo the constant-remat pass.** Unblocked by step 4, and the next quality lever
  after step 2. It failed twice on the seam step 4 removes, never on its policy.
- **Shape A** (`gpr_overshoot=1`, mostly `-O0`, ~8 pairs): real pressure at a call
  with twelve argument operands, where the splitter's overshoot marches forward two
  instructions per round. Step 5 turns it into spill code. Diagnosis in
  `docs/terminator-args-next-steps.md` item 10.
- File sizes. `compile/tests.rs` (3751 lines), `compile/mod.rs` (2043),
  `egraph/algebraic.rs` (2289) are untidy with no evidence of harm, and a rule set
  being long is fine.
