# The CFG should hold VRegs, not ClassIds

The one refactor that unblocks the rest. Written to survive across sessions: it
assumes no context beyond the repo.

## Root cause

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
  reaches them. Anything in lowering that re-resolves a class on its own must
  account for spills and rematerialization — this seam produced seven separate
  wrong-code bugs."*
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

## The fix

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

## What it buys, in order of size

1. **85–94% of block parameters go**, with their copies, their cliques, their slot
   routing, and their per-iteration store/reload traffic. Measured with
   `tests/fuzz/count_trivial_phis.py`; see the table below.
2. **Rematerialization becomes expressible** — the second big lever for
   pressure-bound code, and the reason two attempts at it failed.
3. The splitter's operand rewriting reaches everything, so pressure decisions stick.
4. The bug class that has consumed most sessions disappears, so optimization work
   stops being interrupted by miscompiles.

## Staging

Each step ends green on the full battery. Judge per **(seed, level) pair** with
`tests/fuzz/compare_ref.sh <ref> 60 <shape>`, never on a bare pass count.

| step | work | judged by |
| --- | --- | --- |
| 1 | **Terminator args become VRegs in the CFG.** Completes what `999ae41` started for the schedule. `EffectfulOp::Jump.args`, `Branch.true_args/false_args`. | the 36 shape-B capacity failures (`pressure` at -O1) |
| 2 | **Trivial-phi elimination on those VRegs.** `phi(v, …, v) → v`, self-references ignored, to a fixpoint. Replaces `src/compile/phi_simplify.rs`. | parameters removed; copies removed; the same 36 pairs |
| 3 | **The remaining `EffectfulOp` operands**: `Load.addr`, `Store.addr/val`, `Call.args`, `Branch.cond`, `Ret.val`. | battery + fuzz per step |
| 4 | **Delete the reconstruction machinery** (~312 sites): per-block snapshots, the three-times-patched map, `block_param_vregs`, `block_param_vreg_overrides`, `class_emitted_in`, the `value_defs` guard in `split.rs`. | LOC removed, no behaviour change |

Step 1 is the smallest slice with the largest payoff: the 28-parameter cliques live
exactly there.

### Step 1 notes

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

### Step 2 notes

- The self-reference rule is what reaches loops: a value carried round a loop and
  never reassigned is `p = phi(p_init, p)`, whose operands reduce to `{p_init}`.
  Without it nothing in a loop is ever removable.
- Do **not** re-attempt it on ClassIds. That is what `phi_simplify.rs` does and its
  module doc records both failures: removing the parameter reads a register only one
  path wrote (the machine verifier says so outright), and adding the sound dominance
  condition is too strong to win anything *and* still leaves programs wrong.
- The entry block's parameters are the function's own. Never remove them.

## State when this was written (2026-08-03, `02be4ae`)

Gates: 924 unit, 440 lit at `BLITZ_VERIFY` off/1/strict, 281 differential + `cc`.

Generated corpus, release build, 60 seeds per shape, per (seed, level) pair:

| shape | passing | failures |
| --- | --- | --- |
| `mixed` | 58/60 | 2, both `-O0` |
| `args` | 53/60 | 8, mostly `-O0` |
| `pressure` | 24/60 | 36, **all** `-O1` |

**Every remaining failure on every shape is `register pressure overshoot`.** No
wrong-value program is open and `tests/fuzz/findings/` is empty — a first for this
corpus.

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

## Not part of this refactor

- **A benchmark harness** (ROADMAP P0-Measurement, still unchecked). Worth having,
  and not a prerequisite: steps 1–4 are judged by correctness gates plus the
  capacity numbers above, both of which already exist.
- **Shape A** (`gpr_overshoot=1`, mostly `-O0`, ~8 pairs): real pressure at a call
  with twelve argument operands, where the splitter's overshoot marches forward two
  instructions per round. Separate problem, smaller, and easier to judge once the
  reconstruction machinery is gone. Diagnosis in
  `docs/terminator-args-next-steps.md` item 10.
