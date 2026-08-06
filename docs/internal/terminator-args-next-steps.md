# Register allocation and division: what landed, what is next

> **Partly superseded, 2026-08-05.** Steps 1 through 5 of `docs/internal/refactor-roadmap.md`
> have landed since this was written, and two of its conclusions no longer hold.
> Shape B is *not* blocked by parameters lacking an `Op::BlockParam` marker: with
> every parameter's VReg available (step 3b) the routing pass finds all of them and
> still routes none, because they alias the value they carry and `split.rs`'s
> `value_defs` guard refuses. And the function-scope allocator now has a spill loop,
> which fixed shape A and left shape B untouched. Four attempts on shape B are
> tabled at the end of the refactor roadmap with their numbers; read that before
> acting on anything here.

**For what to do next, read `docs/internal/refactor-roadmap.md`** -- eight ordered structural
steps, tracked. This file is the record of what landed and what was ruled out; the
roadmap is the plan.

All on `master`, pushed. Measured with `bash tests/fuzz/run_fuzz.sh 40 mixed`,
compared per **seed/level pair** — the per-seed view hides one level going
backwards.

| point | fuzz passing | note |
| --- | --- | --- |
| `aeb3789` | 3 / 40 | before the terminator-operand refactor |
| `999ae41` | 27 / 40 | terminator arguments first-class (previous session) |
| `273c7e0` | 34 / 40 | one phi copy per block-param class |
| `34beb3d` | 36 / 40 | live-out-only splitter victims spill cross-block |
| `e115514` | 38 / 40 | conservative (Briggs) coalescing |
| `2b1ec09` | 39 / 40 | division width, grouping and paired extraction |
| `8472e14` | 40 / 40 | a call's result copied out by the barrier's dst |

Since then the shapes are measured separately, 20 seeds each, judged per
(seed, level) pair with `compare_ref.sh`:

| point | mixed | args | pressure | note |
| --- | --- | --- | --- | --- |
| `1e4678f` | 20/20 | 17/20 | 6/20 | two sessions ago |
| `aa91e96` | 20/20 | 17/20 | 5/20 | a parameter that is really a value |
| `021d4ed` | 20/20 | 17/20 | 5/20 | copy pairs from the schedules (one pair) |
| `b5c0667` | 20/20 | 19/20 | 10/20 | the block-parameter shadow |

At 60 seeds per shape, which is where the last two bugs came from:

| point | mixed | args | pressure | note |
| --- | --- | --- | --- | --- |
| `35f99be` | 57/60 | 52/60 | 24/60 | two programs wrong at BOTH levels |
| `0a51f88` | 58/60 | 53/60 | 24/60 | a re-emitted class with no type |

Green at every step, currently 924 unit, 440 lit at off / 1 / strict, 281
differential + `cc`.

**The `mixed` shape is not the whole generator.** Three sessions optimised `mixed`
without knowing `pressure` was at 3/20. Every failure left on any shape is
`register pressure overshoot`: across 120 programs at two levels there is no
wrong-value failure, and what remains is allocator capacity.

## What landed

**Block parameters that share an e-class.** `propagate_block_params` merges a
parameter with its incoming argument when the block has one predecessor and the
argument is constant, so two parameters holding the same constant collapse onto
one class, one VReg and one register. That is sound, and it made
`abi.rs:362` — *parallel copy has a repeated destination* — fire on 7 of 13
failing programs for copies that were exact duplicates. Deduping by canonical
param class in `build_phi_copies` was the whole fix.

**A splitter victim with nothing to split.** A per-block split rewrites the
victim's uses in the block, so it needs a def *and* a use there; the scope choice
tested only for the def. A value defined in the over-pressure block and consumed
downstream — the best victim at the peak, since it holds a register across the
whole block — reached `apply_split_planned`, which found no use and planned
nothing. An empty plan reads as convergence.

**Coalescing made the graph uncolorable.** `coalesce` merged any non-interfering
copy pair. Merging unions two neighbourhoods, so legal merges compound: three
parameters of one block each picked up the range of a value living across the
function and formed a 15-clique against 14 GPRs. The splitter measures the graph
*before* coalescing, where the clique does not exist, which is why it reported
convergence while the colorer needed 15. The Briggs conservative test fixed it.

**Division.** Three bugs, all from one unmodelled fact: an `idiv`'s results live
in RAX and RDX and nothing in the IR says so. The width came from `vreg_types`,
which is built before the splitter runs, and fell back to 64 bits — fatal only
for division, where a negative 32-bit divisor materialized by `mov ecx,imm32`
becomes 4294967293. Barrier grouping pulled a projection past a call that
destroys both registers. And extraction is a *parallel copy*: with the quotient's
destination allocated RDX, `mov edx,eax` destroyed the remainder.

## What is left

### 1. Register pressure, which is now the only failure mode

`pressure` (10/20) and `args` (19/20) fail only by refusing to allocate. Two
mechanisms are already documented and both are still live: the splitter reporting
an overshoot it cannot act on, and the colorer needing more colours than the
splitter's measurement implies. `BLITZ_DEBUG=split` prints each over-budget VReg
with a clique containing it — a clique wider than the budget is proof that
splitting, not a better colouring order, is what is missing.

Start from `tests/fuzz/run_fuzz.sh 20 pressure`, which is the shape built for this
and the one nothing has been tuned against.

**LANDED (2026-08-03).** The phantom register is gone: `Op::TerminatorArgs`,
`StoreBarrier` and `VoidCallBarrier` define nothing, and their `dst` no longer
takes interference edges or counts for pressure (`Op::has_no_result`). It had been
taking a colour at the widest point in the block — `gpr_chromatic=17` against a
budget of 14 with colour 16 held by a `TerminatorArgs` — and the splitter,
measuring the same phantom, stalled at `excess=1 with no eligible victim` for
thirty rounds. `pressure` seeds 11 and 12 now compile at both levels.

It could not land alone: lowering every terminator argument's degree by one let
conservative coalescing admit one more merge, and that merge exposed a
phi-destination bug (see below). Both halves are in, with
`tests/lit/regalloc/phi_dest_from_target_schedule.c` failing without either.

**The wall — mostly down. `pressure` 3/20 → 6/20, ten seed/level pairs, nothing
regressed.** A loop-carried value can now stay in a slot across the loop: block
parameters in excess of their class budget are routed through slots (not only XMM
parameters crossing a call), and an argument whose destination parameter lives in a
slot becomes its own `SpillStore` in the predecessor rather than an operand of
`Op::TerminatorArgs`. Each store goes immediately after the last point its value is
needed for anything else — all of them at the end of the block relieves nothing,
since the value is live until whatever reads it.

Three things that had to be got right, each found by the corpus:

- One VReg can name parameters of **several blocks**, and two parameters of one
  block can be the same e-class. Route the value once, one slot, recording every
  position that names it. Routing per position gave one VReg three slots.
- Drop an argument from the terminator by **the destination it feeds**, never
  because its VReg belongs to some routed parameter — an edge can pass one routed
  parameter's value to a different parameter.
- A spill store defines nothing, so its `dst` must not take a register either.
  Harmless one at a time; not harmless sixteen at a time.

**The wall is down. `pressure` 5/20 → 10/20 and `args` 17/20 → 19/20 in one
change (`b5c0667`), 18 pairs fixed and none regressed.** What was left of it was
one unmodelled fact: a `BlockParam` is a *marker*, and the phi copies on the edge
have written every parameter of the block before its first instruction runs. The
scheduler places the markers by dependence order, so the run of markers is a
region where the schedule says almost nothing is live and the machine has a whole
parallel copy resident. Three parts, and none of them works alone:

- **The colorer** gets an edge from each parameter to everything the block names
  before its last marker. Without this a value that lives and dies in there takes
  a parameter's register and nothing can see it: the register is written, and no
  two *modelled* ranges overlap. This is what `param_shadow_const_v7.c` was.
- **The splitter** counts the parameters as live over that run, or it never sees
  the pressure those edges create.
- **The routing target** becomes the budget less what else the run names. Those
  are the only values the pressure loop cannot help with, because its remedy
  there — spill one, reload it in the same place — just puts another value in the
  region. Everything live further into the block is an ordinary value it *can*
  spill, and deriving the target from those is the collapse recorded below.

Note the earlier failed attempt: modelling parameters as live from block entry in
`compute_local_liveness` **and** `regalloc/liveness.rs` cost 36 → 33 on the mixed
corpus, because extending the allocator's own ranges forces real spill code. The
change that worked touches the splitter's measurement and the interference graph,
never the liveness the allocator spills against.

**The thing to resist, still.** Deriving the routing target from
`budget - other_live_ins` over the whole block: it collapses to "route every
parameter" as soon as the other live-ins fill the budget, which routed 71
parameters in a 40-line test and miscompiled it. The marker run is bounded and a
block's live-in set is not.

**The original wall, for the record.** The remaining overshoot is not something splitting can reach. A
generated `main` has a loop carrying 28 values, so a block has 28 parameters of
which 16 are GPR, and `add_block_param_interferences` makes them a 16-clique
against 14 registers. What the splitter does with that is spill each parameter
and then reload all sixteen immediately before the `TerminatorArgs` that feeds
them back, which recreates the clique one instruction later; from round 13 on it
reports `excess=1 with no eligible victim` at exactly those points, every victim
being a reload it inserted itself.

So a loop-carried value has to be able to **stay in a slot across the loop**.
The machinery is already there and only reachable for one case:
`split::detect_blockparam_call_crossings` slot-routes XMM parameters that cross
a call, `lower_terminator` emits the predecessor's store, `apply_plan_to`
truncates the parameter's segment so no register is allocated, and
`remove_terminator_arg_operands` takes the argument out of the parallel copy's
clique. Generalising the trigger from "XMM and crosses a call" to "parameters of
either class in excess of the class budget" is the shape of the fix. The one
piece that is not there: a slot-destined argument still needs a register at the
terminator, because `PhiCopy::Slot` stores from `src_reg`. Making it its own
`SpillStore` instruction in the predecessor's schedule instead of an operand of
`TerminatorArgs` is what keeps those stores from being one simultaneous clique —
a store to a slot clobbers no other argument's register, so they can be
sequential.

### 2. Delete the Phase 7 coalesce-alias patching — DONE

`compile/mod.rs` built `block_class_to_vreg` by collapsing each class to one VReg
and overlaying segments. It now renames each segment through coalescing in place,
keeping every range: the full-range entries linearization made still answer a
lookup at a point no narrower segment covers, and nothing fabricates one. What had
forced the collapse was the inverse index — coalescing makes one VReg serve several
classes, which one-class-per-VReg cannot express and `insert_segment` asserts — so
`insert_segment_shared` now says outright that such a map carries only the forward
direction, and `registered_class`, which existed only to dodge the assertion, is
gone.

Landing it also let the remaining consumers shrink to fallbacks: a load's
destination and a call's result come from the barrier's `dst`, a folded address and
a `Ret`'s value from the VReg the schedule names, and Phase 7's barrier grouping —
118 lines whose output nothing read — is deleted. A `Ret` that cannot name a
register is now an error rather than a silently omitted move.

**It did not fix `briggs_admits_illegal_merge.c`**, which was the stated
expectation and was wrong. That bug turned out to be the phi *destination*: a block
parameter can have two VRegs, and `build_phi_copies` asked the class while the
target block read what its own `BlockParam` instruction carried. Fixed by asking the
target block's schedule first.

What is left of this item: the class map is now a fallback everywhere in Phase 7.
Deleting it outright needs the cases where a role operand is absent
(`populate_effectful_operands` returning early on an empty `vregs`, or an op with
no barrier at all) to be handled, and measurement of whether they are reachable.

This session added a third reason to do it. Every wrong-code bug fixed here was
the same shape — a pass resolving a *class* where it should have used the VReg the
schedule carries — and each fix replaced one class lookup with the schedule's own
answer: terminator arguments, a division's projections, and a call's result. The
remaining class-resolving consumers are `build_barrier_context` and the
effectful-op operand resolution.

It also blocks *verification*. A slot-level verifier — a reload must produce the
class that was stored — cannot be built on that map: it reports false positives on
the splitter's own immediate store/reload pairs, because the map has already
collapsed each class to one VReg. That check is worth having once the map is
trustworthy.

And it is where `briggs_admits_illegal_merge.c` points. Every liveness the
compiler computes says the five VRegs that merge there are live in **disjoint
blocks** — read it off with the new `BLITZ_DEBUG=liveness` dump, which now covers
the function-scope allocator and not only the per-block one: `162` in blocks
38/41/76/77, `169` in 91, `174` in 78–83, `190` in 80/84, `175` in 85, no two
ever together. So the interference graph is right about the pair and the merge is
legal by the model; what breaks is downstream of it, where five classes now
resolve to one VReg. Next probe: R14 is written by a six-way phi rotation at an
edge that is *not* one of this web's edges (`mov %r15d,%r14d`, and R14 holds 62
where p4's 0 belongs), so find which block's parameter also holds R14 there and
whether the two ranges overlap in the emitted code but not in the model.

### 3. `main` falling off its end — DONE

C99 5.1.2.2.3 says the implicit return is 0; blitz returned whatever was in EAX
(8 at `-O0`, 223 at `-O1`). Correct at both levels, and now
`tests/lit/control/main_falls_off_end.c`.

### 4. The register-sharing violations — DONE, and they were the checker's

Every one of them, in all four remaining reproducers, was
`verify_register_sharing` propagating a successor's block parameters into its
predecessor's live-out. The allocator does not: a parameter is written by the phi
copy on the edge, so what is live at the predecessor's exit is whatever its
terminator passes, which `phi_uses` already carries. The check now models
parameters the same way, via
`compute_global_liveness_with_block_params`.

The shape that made it fire: an `Iconst(0)` class whose VReg is *also* the
parameter VReg of a later block, so the def in block 0 is dead — the value the
block really passes is a second copy of the class emitted at the terminator.
Under the old model that dead def read as live to the end of the block, and
everything holding its register in between read as a clash. Two reports came out
of one function that way.

Read off with `BLITZ_DEBUG=liveness`, which now prints each block's successors
beside its live sets. `block 0: succs=[65] live_out=[...]` against
`block 65: params=[1, ...]` says in one line that the parameter is why VReg 1
left block 0's live-out, and the checker's disagreement with that is the bug.

All four programs became lit tests (`control/main_falls_off_end.c`,
`regalloc/folded_addr_under_pressure.c`, `regalloc/mixed_pressure_seed12.c`,
`regalloc/float_sum_guard_seed23.c`).

### 5. The `pressure` shape's wrong-code bug — FIXED (`f5ab27a`)

`tests/lit/regalloc/shared_vreg_two_arg_positions.c`, from seed 18: cc 106, blitz
-O0 82, with exactly one term of the sum wrong.

**One VReg filling two argument positions.** An edge whose target has two
parameters of the same e-class passes one VReg twice. Position 0's destination
was routed through a slot, position 15's was not, and
`remove_terminator_arg_operands` dropped operands *by VReg* -- so position 15
lost its operand as well, no phi copy was emitted for it, and the target block
read whatever the register held. The comment above the caller already stated the
rule (decide per argument, by the destination it feeds) while the code below it
built a set of VRegs. Removal now takes argument indices.

`pressure` seed 7 at -O0 goes from exit-nonzero to passing; nothing regressed on
any shape, measured per seed and level with `compare_ref.sh`.

Two dead ends on the way, both worth not repeating. An **illegal coalesce** looked
certain -- the phi trace prints block 18's parameter 15 as a VReg holding RAX,
which reads as a parameter sharing a register with a live value -- but the trace
chases coalesce aliases before printing, and that VReg's own range is entirely
inside one block. And a **frame-layout overlap** looked certain when a slot
appeared to change value with no store to it; that was `read_frame.py`
mislabelling readings, since it attached them to a fixed sequence of `continue`s
while the address sat in a loop.

Still open on this program: -O1 cannot allocate it (`gpr_overshoot=3`), which is
why the test is pinned to -O0.

### 6. Coalescing merged onto a VReg from the other arm — FIXED (`021d4ed`)

`tests/lit/regalloc/coalesce_pair_from_schedule.c`, seed 18 again: cc 442, blitz
-O0 432, one term wrong -- `v11 = (((v13 / 7) & 1023) | 11)` arriving as 1, and an
OR with 11 cannot produce 1.

**`compute_copy_pairs` asked the function-wide map.** A pure class is re-emitted in
every block that needs it, so a lookup at block 7's exit returned the VReg block
16 defines -- the two blocks are on opposite arms of a branch. Coalescing took that
as a legal merge candidate and merged block 15's parameter onto it, so the
parameter's register was one picked for a value on the other path. v11 is
loop-carried and slot-routed, and the latch's store into its slot then read that
register: the loop counter. The pairs now come from the argument's
`Op::TerminatorArgs` operand and the target block's own `BlockParam`, the same two
sources `build_phi_copies` uses, and are computed after the splitter so a
slot-routed argument contributes none.

**Why nothing caught it.** Coalescing decides only which VRegs share a register;
every copy around the merge stayed self-consistent, and both verifiers hold --
def-before-use is satisfied and no two *modelled* ranges overlap. The chain was
read off the running binary one store at a time (`read_frame.py`,
`read_double_sum.py`), then the divergence was named by comparing what
`append_terminator_args` recorded for an argument against what `compute_copy_pairs`
resolved for the same argument at the same point: v628 against v474.

`pressure` seed 18 at -O0 fixed; `mixed` 20/20, `args` 17/20 and the rest of
`pressure` unchanged, measured per seed and level. -O1 still cannot allocate this
program, so the test is pinned to -O0.

The reduction is a **different** bug, and is now
`tests/lit/regalloc/param_shadow_const_v7.c` -- see item 1.

### 7. Redo the constant-remat pass

Implemented, measured and reverted twice, both times on the Phase 7 seam rather
than the policy. Item 2 is the precondition.

### 8. The findings directory turned over

All four earlier reproducers are lit tests (`regalloc/param_shadow_const_v7.c`,
`param_shadow_seed13.c`, `param_shadow_seed16.c`, `param_shadow_seed3.c`), and all
four were the same bug -- the block-parameter shadow of item 1. Three pass at both
levels and stay unpinned so `run_diff.sh` covers them at `-O0`, which is what
caught them: `run_tests.sh` only exercises tinyc's default `-O1`.

### 9. Two programs wrong at BOTH levels — FIXED (`9207141`, `0a51f88`)

Widening to 60 seeds per shape found them; at 20 there was no wrong-value program
left. Both were **one bug**, and the signature was new: every bug of the previous
three sessions was wrong at one level and right or uncompilable at the other, so
`-O0`-vs-`-O1` carried them. These were equally wrong at both, which only the
reference compiler sees.

**A flags-only 32-bit compare was emitted at 64-bit width.** `cmp r8,rdi` where
`mov edi,-2` had zero-extended, so `14 < -2` compared 14 against 4294967294 and
came out true; `if ((v2 < v6)) { arr[((58 - v2)) & 7] = 81; }` then ran with
v2=14, v6=-2 and `(58 - 14) & 7` is 4, so 81 landed on arr[4].

The width came from `vreg_types`, missing for that VReg. **A class re-emitted in a
later block gets a VReg of its own, and the restore after each block in
linearization is an `insert_single` -- which replaces the class's segments, so the
function-wide map keeps one re-emission and the others have no type at all.**
`vreg_types` is now built from the per-block snapshots too, and the dead-difference
compare takes its width from the operands rather than the dst, since the dst of a
flags-only sub names a value nobody materialises.

Note the shape it needed: the compare must reach lowering's dead-difference path
*while the same subtraction is also live*, because `apply_icmp_isel` shares one
`X86Sub` between `Icmp(cc, a, b)` and `Sub(a, b)` -- so the class is emitted once
for its difference, with a type, and once for its flags in another block, without
one. `v2 < -2`, `v6 > v2`, and dropping the live `v2 - v6` all compile correctly.

Test `tests/lit/regalloc/cmp_width_reemitted_class.c`, 13 lines, beside
`div_width_from_op.c` -- the same hole, found in division first, closed there by
putting the width on the op. 3 pairs fixed over 60 seeds per shape, none regressed.

### 10. The remaining 46 failing pairs, all capacity

`mixed` 58/60, `args` 53/60, `pressure` 24/60 with the release build. All 46
failing pairs are `register pressure overshoot` -- **not one wrong-value failure on
any shape**, which the corpus has never shown before. `tests/fuzz/findings/` is
empty. Do not widen the generator past 60 seeds until these are fixed.

They come in two shapes, and the split is by level:

**Shape A -- `gpr_overshoot=1`, mostly at `-O0`** (8 pairs: `args` 3, 31, 45, 49,
51, 60; `mixed` 24, 57). The smallest is `args` seed 3 at 79 lines. The splitter
does *not* stall: it finds the overshoot and acts, and the overshoot **marches
forward two instructions per round** -- `[169] → [171] → [173] → [175]` -- with the
same victim every time, until the round limit. Each round cross-block-spills that
victim to a *fresh slot*, so one value ends up with four slots and four stores.
The peak instruction is a call with twelve argument operands, so most of what is
live there cannot move.

**Shape B -- `gpr_chromatic` 16-18 against 14, all at `-O1`** (36 pairs, the whole
`pressure` shape; `-O0` is 60/60 there). The splitter *stalls*:
`excess=1 with no eligible victim`, and every candidate is a `SpillLoad` it
inserted itself, in a run of them immediately before a `TerminatorArgs` carrying 28
arguments. Spill-everything-then-reload-everything recreates the clique one
instruction later.

For shape B the relief mechanism exists and is blocked. The 28-argument edge feeds
a **single-predecessor** block, and 16 of that block's 28 parameters have no
`BlockParam` marker at all: linearization's middle branch skips a parameter whose
class is already emitted when the block has one predecessor, because the class map
names the value everywhere at that point. `detect_blockparam_slot_routing` finds
parameters through `find_block_param_vreg`, which needs a marker, so those 16 GPR
parameters are **invisible to routing** and nothing can relieve them. Blocks whose
parameters all have markers (b1, b7 in `pressure` seed 22) get 17 routed each.

**The answer is neither of the two mechanisms below: it is to stop creating the
parameters.** Measured with `tests/fuzz/count_trivial_phis.py`, which reads
`--emit-ir` and needs no compiler change:

| program | block parameters | redundant |
| --- | --- | --- |
| `pressure` seed 22 -O1 | 97 | 88 (90%) |
| `pressure` seed 5 -O1 | 197 | 186 (94%) |
| `pressure` seed 7 -O1 | 85 | 79 (92%) |
| `mixed` seed 58 -O1 | 133 | 120 (90%) |
| seed 18 (the 235-line lit test) -O0 | 247 | 212 (85%) |

Per block on seed 22: the loop header carries 28 parameters of which **4** are real,
block 7 carries 28 of which 3, and block 20 carries 28 of which **none** -- it is a
single-predecessor pass-through block whose entire parameter list is redundant, and
its incoming edge is the 28-argument terminator that stalls the splitter.

A block parameter is a phi, and `phi(x, ..., x) -> x` with self-references ignored is
Braun et al.'s `tryRemoveTrivialPhi` -- the step `read_variable` never had. It creates
a parameter before recursing, to break loop cycles, and keeps it whatever the operands
turn out to be. The self-reference rule is what reaches loops: a value carried around
a loop and never reassigned is `p = phi(p_init, p)`, whose operands reduce to
`{p_init}`.

This is the code-quality answer and not merely the capacity answer. Each removed
parameter removes a register-to-register copy per incoming edge per iteration, a place
in the parameter clique, and -- where the splitter had routed it -- a store and a
reload per iteration for a value nothing reads. It is a canonicalization: LICM, DCE,
coalescing and the splitter all get simpler input, and it *removes* special cases
(the `value_defs` guard, the `block_param_vregs` backup, the marker-less parameter
case) rather than adding another mechanism. Both of the mechanisms below are ways of
coping with parameters that should not exist, and the first would make code quality
*worse*: it adds a real copy per pass-through parameter.

**Started, behind `--enable-phi-simplify`, off by default** (`src/compile/phi_simplify.rs`,
`02be4ae`). What is not yet right, and it is the whole difficulty: **one e-class is one
*expression*, not one value.** Two predecessors computing the same expression share a
class while each emits it into a register of its own, which is exactly what the phi
reconciles; removing it leaves the block reading a register only one path wrote, and
the machine verifier says so -- "reads XMM8 on a path where nothing writes it". Adding
the sound dominance condition (every supplying predecessor dominates the block) is too
strong to win anything -- seed 22 returns to `gpr_overshoot=4` -- and two programs still
come out wrong at -O1 with both verifiers silent, so dominance is not sufficient
either. The condition has to express "the value is in one register on every path that
reaches this block", which is a fact about the *extraction*, not the CFG. Likely
shapes: remove the parameter only where the class is not re-emitted per block
(`class_emitted_in` and the per-block snapshots already track that), or run the pass
after extraction and pay for rebuilding everything keyed on parameter index.

`EGraph::rewrite_block_params` is done and is the reusable half: it renumbers
`BlockParam` nodes in one pass over a drained memo, because the new index of a
surviving parameter is usually the old index of a removed one and any incremental
rewrite collides.

The two mechanisms that cope instead, kept for the record:

- **Mint a real parameter for every position** so slot routing sees all of them. The
  "principled alternative" the `aa91e96` note records. Adds a copy per pass-through
  parameter, so it is the wrong direction for code quality.
- **Drop the terminator operand where the argument's class equals the parameter's
  class**, and stop treating that VReg as a parameter for liveness. Subsumed by
  removing the parameter outright, which is the same idea done in the IR.

## Ruled out — do not repeat

- **A better colouring order for the `gpr_overshoot=1` shape.** The over-budget
  report names a clique of 8 against a budget of 14, which reads as a colouring
  artefact and is not one: reverse MCS, descending degree, and a full
  Chaitin-Briggs simplify/select (remove below-budget nodes first, take the widest
  optimistically when none is left, colour in reverse) all need exactly 15 colours
  on `args` seed 3. The clique in that report is a greedy lower bound, not the
  maximum. The splitter had already *measured* `gpr=15` at that point and acted on
  it, so the constraint is which victim it picks, not how the graph is coloured.
- **Blocking a victim the splitter already spilled in an earlier round.** One value
  getting four slots and four stores across four rounds looks like pure waste and
  is not: each round's schedule has the earlier round's uses rewritten to reloads,
  so re-spilling the original spills a different remaining part of its range.
  Seeding `planned_victims` from the previous rounds made the corpus *worse*
  (`args` seed 3 overshoot 1 → 3, `mixed` seed 24 1 → 2) while helping one
  `pressure` seed. Reverted.

- **Modelling a block's parameters as live from block entry in the allocator's own
  liveness** (`regalloc/liveness.rs`), which is how the first attempt at this did
  it, together with the splitter's `compute_local_liveness`. Extending the ranges
  the allocator spills against forces real spill code: a net loss on the corpus,
  36 → 33, with a new pressure failure and a *behaviour* regression where `-O0`
  and `-O1` disagreed. Modelling it in the splitter's measurement and the
  interference graph *only* is what worked (`b5c0667`, item 1).
- Excluding slot-spilled or splitter-truncated VRegs from coalescing (two earlier
  attempts, both reverted). Conservative coalescing is a different mechanism and
  is what actually helped.
- **A slot-owner verifier** ("two values stored into one slot"). Sound but
  vacuous: the splitter allocates a slot per victim and every slot in the program
  that needed the check is stored exactly once.
- **Widening `add_call_precolors_for_block`'s `call_count == 1` guard** so every
  call result is pinned to the ABI return register. Two results pinned to one
  register collide whenever their ranges overlap; copying the result out is the
  fix, not pinning it.

## Ground rules that held up

- Judge regressions per seed/level pair, never per seed.
- Do not ship a change that is net-negative on the corpus even when it fixes the
  target program.
- One change at a time, with the full battery between them.
- A "did not compile" failure is not a wrong-code failure. Categorise before
  reading a wall of red as a verdict.
- `TINYC=<path> bash tests/lit/run_tests.sh` proves a new regression test fails
  before the fix. tinyc defaults to `-O1`, so an `-O0`-only bug needs
  `// FLAGS: -O0` — which excludes the test from `run_diff.sh`, by design.
- Measure every generator shape, not just `mixed`. Three sessions optimised one
  shape without knowing `pressure` was at 3/20.
- When a program is near the allocator's limit, **a probe that adds an
  instruction moves the bug**. Read values with `tests/fuzz/read_double_sum.py`,
  which keeps the instruction count fixed, or with gdb on the unmodified binary.
