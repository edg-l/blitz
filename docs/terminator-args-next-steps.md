# Register allocation and division: what landed, what is next

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

Green at every step: 917 unit, 414 lit, 270 differential + `cc`, `BLITZ_VERIFY`
off / 1 / strict.

**The `mixed` shape is not the whole generator.** Measured for the first time:
`args` is 17/20 and `pressure` is 3/20. Every one of those 38 failures is
`register pressure overshoot` (plus one `Load: no register for result`) — so
across 80 programs at two levels there is now **no wrong-value failure at all**,
and what is left is allocator capacity.

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

`pressure` (3/20) and `args` (17/20) fail only by refusing to allocate. Two
mechanisms are already documented and both are still live: the splitter reporting
an overshoot it cannot act on, and the colorer needing more colours than the
splitter's measurement implies. `BLITZ_DEBUG=split` prints each over-budget VReg
with a clique containing it — a clique wider than the budget is proof that
splitting, not a better colouring order, is what is missing.

Start from `tests/fuzz/run_fuzz.sh 20 pressure`, which is the shape built for this
and the one nothing has been tuned against.

**Measured on the `pressure` shape (all 17 failures, 2026-08-03).** Two things,
one small and one that is the whole wall.

*The small one, and it is fixed but not shippable.* `Op::TerminatorArgs` has a
`dst` that names nothing — lowering skips the op — and
`build_interference_into` still gives that `dst` an edge to every value live at
the terminator, so a phantom takes a colour at the widest point in the block.
`BLITZ_DEBUG=split` shows it directly: `v369 GPR color=16 ... def=b5
TerminatorArgs(...)` with `gpr_chromatic=17` against a budget of 14. Excluding
it (`Op::has_no_result()`, honoured in the def-interference loop and in
`compute_pressure_for_class`) is worth `pressure` seed 12 at both levels and is
neutral on `mixed`, but it regresses `args` seed 9 `-O1` to a **wrong answer**,
so it must not land alone. See
`tests/fuzz/findings/briggs_admits_illegal_merge.c`: lowering every terminator
argument's degree by one lets Briggs admit one more merge, and that merge is
wrong. The patch is written out in that file's header.

*The wall.* The remaining overshoot is not something splitting can reach. A
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

### 2. Delete the Phase 7 coalesce-alias patching

`compile/mod.rs` builds `block_class_to_vreg` by collapsing each class to one VReg
and overlaying segments — documented as a landmine under P0 in `ROADMAP.md`.

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

### 3. `main` falling off its end

`tests/fuzz/findings/main_fallthrough_exit_status.c`. C99 5.1.2.2.3 says the
implicit return is 0; blitz returns whatever is in EAX (8 at `-O0`, 223 at `-O1`).
A six-line `main` is correct, so it takes surrounding code.

### 4. A latent register-sharing violation

`BLITZ_VERIFY=strict` on the seed 7 reduction reports *VReg 9 and VReg 1056 are
both live and both hold RBX* at block 0's exit, at `-O1`. Predates this work and
the program prints the right answer, so it is latent rather than wrong-code.

### 5. Redo the constant-remat pass

Implemented, measured and reverted twice, both times on the Phase 7 seam rather
than the policy. Item 2 is the precondition.

## Ruled out — do not repeat

- **Modelling a block's parameters as live from block entry**, in the splitter's
  `compute_local_liveness` and/or `regalloc/liveness.rs`, to match what
  `add_block_param_interferences` asserts. Arguably the honest model, and it does
  add real edges, but it is a net loss on the corpus: 36 → 33, with a new pressure
  failure and a *behaviour* regression where `-O0` and `-O1` disagreed.
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
