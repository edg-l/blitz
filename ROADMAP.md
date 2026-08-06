# Blitz Roadmap

Supersedes the old `goals.txt`, `next.txt`, and `ideas.txt`.

## Goal

Produce the best possible x86-64 machine code. Blitz targets **one** ISA on
purpose: the optimizer can reason natively about addressing modes, flags, LEA,
multi-output instructions, and uarch behavior instead of laundering everything
through a target-neutral IR. Anything a portable backend has to generalize away
is fair game here.

The measure of success is code quality against `gcc -O2` / `clang -O2` on the
same input: fewer instructions, fewer bytes, fewer spills, better loops.

Correctness is a precondition, not a tradeoff against that. An aggressive
single-target optimizer has more room to be subtly wrong than a conservative
portable one, so the correctness infrastructure in P0 is part of the goal
rather than overhead against it.

## Non-goals

- **tinyc is not a product.** It exists to feed the backend real-ish input and
  to make tests readable. Frontend work happens only when a missing C feature
  blocks a codegen experiment we want to run. Never for its own sake.
  Explicitly not planned: preprocessor, typedef/enum/union, initializer lists,
  function pointers, bitfields, diagnostics quality.
- **Multi-target.** Single-ISA specialization is the whole thesis.
- **Toolchain completeness.** DWARF, `.eh_frame`/CFI, PIC/PIE, TLS, inline asm,
  atomics, sanitizers, LTO, PGO. These are what a *shipping* compiler needs, not
  what a *good* optimizer needs. Revisit only if one blocks measurement.

## Start here

Ordered. Each says what it is, why it is placed there, and what would tell you it
is done.

1. **Fix the four open miscompiles.** `mixed` seed 109, `args` seeds 52 and 175,
   `pressure` seed 128, 131, 158 or 165, with reproducers at
   `~/.cache/blitz-fuzz-fails/`. Correctness is a precondition of this project's
   goal, not a competing priority. All four are right at `-O0` and wrong at
   `-O1`, which narrows where to look: the passes `-O0` does not run, or the ones
   whose input `-O0` leaves small. Separately, `mixed` 57 is a **capacity
   regression** step 6's fold introduced; it is the one open failure this tree
   caused rather than inherited. Done when a 200-seed run of each shape is clean.
2. **Make the gate able to see them.** `run_fuzz.sh` is pinned at 30 seeds
   everywhere, and at 30 seeds all three shapes are green while four programs
   miscompile. **A session can work all day, see every gate pass, and never learn
   that.** Widening costs 67s a shape; a saved corpus of known-failing programs
   would make the routine check seconds. Do this alongside 1, or 1 has no oracle.
3. **Give LICM a pressure check.** The largest measured quality gap in the tree:
   `BLITZ_PASSES=-licm bash tests/run_codesize.sh` takes `bench` reloads from 377
   to 149 and spills 84 to 30, and *lowers* instructions 2637 to 2518. It hoists
   every invariant it can prove, and a value hoisted out of a loop is live across
   the whole body. Done when `-O1` beats `-O0` on all 15 `bench` kernels.
4. **Offset-aware alias analysis.** ~50 LOC in `alias.rs`, and the cheapest
   quality win available: today any write to a base invalidates every cached load
   at that base, so `s->a` and `s->b` kill each other, which throttles the
   forwarding and DSE passes already shipped.

After those, the P1 list below is ordered by measured impact. **`P2` is where the
single-target thesis is supposed to pay off and none of it is started** --
immediate-form ALU ops are shovel-ready and mirror the shipped `X86CmpI`.

**Do not start in the register allocator, the splitter, or the block-parameter
machinery without reading `docs/refactor-roadmap.md` first.** It is finished as
work and is now the record of what eleven steps measured -- including the
predictions that were wrong, which is what stops the next attempt repeating them.
Three are worth knowing before touching the splitter: the Chaitin ratio has been
rejected by measurement three times, `insert_early_barrier_spills` cannot be
gated on pressure because it runs before global liveness exists, and
`Op::BlockParam` cannot leave the e-graph because expressions consume it.

## Current state (2026-08-06)

- 925 Rust tests + 480 lit tests, all green. `cargo fmt` clean, zero build warnings.
- `BLITZ_VERIFY=1` and `BLITZ_VERIFY=strict` green across both suites.
- `bash tests/lit/run_diff.sh`: 302 tests compared O0-vs-O1 and against a
  reference compiler; no skips, no differences under gcc or clang.
- Generated programs at 30 seeds a shape -- the width every gate runs -- are
  `mixed` 30/30, `args` 30/30, `pressure` 30/30. **That width measures nothing.**
  At 200 seeds it is `mixed` 195/200, `args` 185/200, `pressure` 189/200: **4
  wrong-value programs and 27 capacity failures.** The 30-seed run is green
  because it is too narrow, not because the compiler is correct. See Known bugs,
  and item 2 of Start here.
- Code quality has a baseline: `bash tests/run_codesize.sh --check`, 894 rows
  across `lit`, `bench` and `fuzz`. **`-O1` emits worse code than `-O0` on 7 of
  the 15 `bench` kernels**, and LICM is 60% of it -- see P1 below.
- **Compile time is quadratic in blocks x classes.** `secs ~ (B*C)^0.86`,
  R2=0.92 over 44 (program, level) points, and both levels sit on one line, so
  `-O1` is not intrinsically cheaper -- it just hands the same pipeline a smaller
  IR. DCE is what shrinks it: `args` seed 108 takes 33s at `-O0` and 4.8s with
  `BLITZ_PASSES=+dce`, and `-O1 -dce` is slower than `-O0`. The two Theta(B*C)
  loops are `linearize`'s per-block evict-and-restore of the whole class map and
  the splitter's pressure scan. `bash tests/profile.sh <src> [flags]` is the way
  in; `perf report` hangs on these profiles, `perf script` does not.
- Pipeline: IR -> inlining -> DCE1 -> store/load forwarding -> DSE -> LICM ->
  e-graph saturation -> cost-based extraction -> DCE2 -> linearize -> trivial
  block-parameter removal (re-extract + linearize again) -> DAG schedule ->
  live-range splitter -> function-scope Chaitin-Briggs regalloc -> terminator
  lowering -> MachInst lowering -> branch relaxation -> ELF.
- Implemented e-graph rules: see `docs/egraph-optimization-roadmap.md`.
- Splitter design: see `docs/split-pass-plan.md`.

## Priorities

### P0 -- Correctness

A fast wrong answer is worthless, and an optimizer this size certainly has
subtle bugs left. Every miscompile of the last several months was found by the
generated corpus and its two oracles, not by a hand-written test happening to hit
the right shape -- which is the argument for keeping the generator ahead of the
compiler rather than treating a green suite as evidence. The priors on where bugs
live: **regalloc** (splitting, coalescing, spill/remat, cross-block liveness)
first by a wide margin, then the **memory passes** (forwarding/DSE resting on a
conservative alias model), then **isel width and type handling** (the `X86CmpI`
`ty` bug was exactly this class).

**What exists**, all green and described in `CLAUDE.md`: the `-O0`-vs-`-O1`
differential with a `cc` oracle beside it (`run_diff.sh`), the UB-free generator
and its shrinker (`gen_c.py`, `reduce.py`), the per-pass IR verifier and its
strict mode (`BLITZ_VERIFY`), and the machine-level verifier over the final
instruction stream. The gate set is **fixed at four runs**; new invariants go
inside the runs that already happen. A battery that grows every time something is
learned stops being run between every change, and one-change-at-a-time is what
makes attribution possible here.

- [ ] **The allocator's liveness disagrees with the emitted code's.** What
      `verify::verify_register_sharing` points at now that it is in.
      `build_interference_into` adds an edge for every simultaneously-live pair,
      so two VRegs could only share a register if the allocator's liveness never
      had them live together -- while liveness recomputed from the emitted
      schedules does. It flags 3 of 40 fuzz programs. Neither VReg in the seed-20
      report is pre-colored, so it is not a pre-coloring artifact.

      Related hole worth an assertion regardless: `greedy_color` and
      `interval_color` apply pre-colorings unconditionally, without checking that
      two of them sharing a color do not interfere.
- [ ] **Callee-saved registers actually preserved.** The one machine-level
      property `MachInst::defs()`/`uses()` do not yet carry.
- [ ] **A stronger UB guard in `reduce.py`.** It does not know about undefined
      behaviour the reference compilers agree on by luck, which is how it deleted
      the array initialisers and had to be corrected by hand. A third compiler, or
      the generator re-simulating the candidate, would close it.
- [ ] **Reference IR interpreter.** The stronger oracle: execute the IR
      directly and compare against the compiled binary. Also lets a failure be
      attributed to a specific pass by re-running the interpreter on the IR
      after each stage.
- [ ] **Rewrite-rule equivalence tests.** For each algebraic/strength rule,
      randomized equivalence check of LHS vs RHS over the operand space
      (including boundary values: 0, 1, -1, INT_MIN, INT_MAX, wraparound).
      Cheap, and it is the only systematic defense against a rule that is right
      for most inputs. The two rejected signed-ordering rewrites are the
      cautionary example.
- [ ] **Regalloc stress mode.** Generate programs with tunable register
      pressure, live-range-crossing-call density, and phi-heavy control flow,
      then differential-execute. Aim it at the code with the worst historical
      bug rate.
- [ ] **csmith-lite for tinyc**: random C restricted to the parseable subset,
      differential against `gcc -O0`/`clang -O0` on the same source. Covers the
      frontend-to-backend seam that IR-level fuzzing skips.
- [ ] **Failures become tests, permanently.** Every fuzz find lands as a lit
      test. Per `CLAUDE.md`, a committed failing test that reproduces a real bug
      is more valuable than a green suite.

### P0 -- Measurement

Without numbers, "most optimized" is unfalsifiable and every session drifts.

**What exists**: `bash tests/run_codesize.sh [--check|--update]` over three
corpora with a baseline each in `tests/baselines/`, fed by `BLITZ_DEBUG=stats`.
Instruction count, `.text` bytes, spill stores and reloads per (program, level);
a generated program that does not compile is a `-` row rather than an omission,
so the holes stay visible.

- [ ] Wall-clock via hyperfine, and the same table for `gcc -O2` / `clang -O2`
      beside blitz's own numbers. Deferred deliberately: blitz-against-itself is
      what refereed the decisions so far, and an external column moves when the
      system compiler updates.

### P1 -- Optimizer gaps with the largest measured impact

- [ ] **LICM has no pressure check, and it is the largest measured quality gap
      in the tree.** Measured on the `bench` corpus at `-O1` with
      `BLITZ_PASSES=-licm bash tests/run_codesize.sh`: turning LICM off takes
      reloads from **377 to 149** and spills from **84 to 30**, and *lowers* the
      instruction count from 2637 to 2518. Per kernel it is a trade the pass
      always takes and often loses -- `matmul` 172 insts / 23 reloads becomes
      163 / 0, `binary_search` 136 / 39 becomes 113 / 0, `hash_table` 415 / 150
      becomes 340 / 45 -- against a genuine saving of 2 to 14 instructions on
      the six kernels where the hoisted value fits (`sieve`, `crc32`,
      `bitcount`, `dot_product`, `nbody_step`, `struct_walk`). The pass hoists
      every invariant it can prove, and a value hoisted out of a loop is live
      across the whole body: on a loop whose body already needs most of the
      register file that is a spill and one reload per use. What it needs is a
      hoist decision that consults the same pressure the splitter measures,
      which is why this sits behind the refactor rather than in front of it --
      `docs/refactor-roadmap.md` step 5 gives the allocator a spill loop and
      step 2 changes how many values are in flight, and both move the number
      this policy would be tuned against.
      Second contributor, same corpus: `BLITZ_PASSES=-inlining` takes reloads to 247
      and insts to 2422, so inlining is paying for itself in call overhead and
      losing in pressure. Same fix shape, same reason to wait.
- [ ] **Offset-aware alias analysis.** Today any write to a base invalidates
      every cached load at that base, so `s->a` and `s->b` kill each other.
      Byte-offset + width disjointness in `src/compile/alias.rs` is ~50 LOC and
      directly unlocks the forwarding and DSE passes already shipped.
- [ ] **GVN / cross-block CSE.** The e-graph does local CSE only. Repeated
      address computations and field loads survive across blocks. Typical
      5-15% on real code.
- [ ] **Loop strength reduction + induction variable recognition.** Every array
      loop recomputes `base + i*scale`. Worth 2-5x on loop-heavy code and is
      table stakes for calling the backend "optimizing".
- [ ] **SCCP.** `propagate_block_params` only handles single-predecessor
      constants; conditional arms that become constant are missed.
- [ ] **Memory SSA / memory versioning.** Makes forwarding, DSE, and GVN work
      cross-block on shared machinery instead of three intra-block passes.
- [ ] **nsw/nuw/nnan/ninf op flags.** Without them the signed-ordering
      algebraic rewrites stay permanently rejected (see Decisions). Op-flag
      bitfield threaded through saturation.
- [ ] **Tail call optimization.**
- [ ] **Loop unrolling.** Compounds with LSR; do it after.
- [ ] **Narrowing / type-width analysis.** `(uint8_t)x + 1` should not promote
      to i32. Domain: `(min_bits, signed)` per e-class.
- [ ] **Dead call elimination** for provably pure functions.

### P2 -- x86-64 specialization (the differentiator)

This is where single-target focus is supposed to pay off. LLVM has ~10x the
isel patterns; we should beat it on the ones we implement.

- [ ] **Immediate-form ALU**: `X86AddI`/`X86SubI`/`X86AndI`/`X86OrI`/`X86XorI`
      for Iconst RHS, mirroring the shipped `X86CmpI`. Shrinks every `x + 1`,
      `x & 0xff`, `x | 8`. Shovel-ready; named as next step by the 04-23 session.
- [ ] **Bit instructions**: `bt`/`bts`/`btr`/`btc`, `popcnt`, `bsr`/`bsf`,
      `tzcnt`/`lzcnt`, `bswap`.
- [ ] **BMI/BMI2 when available**: `andn`, `bextr`, `blsi`/`blsr`/`blsmsk`,
      `shlx`/`shrx`/`sarx` (no flag clobber, no CL constraint), `mulx`.
      Needs a CPU feature level knob.
- [ ] **Carry-chain forms**: `adc`/`sbb`, and `setcc`-free branchless idioms
      (`sbb reg,reg` as a 0/-1 mask).
- [ ] **Rotates and double shifts**: `rol`/`ror`, `shld`/`shrd`.
- [ ] **LHS-iconst compare commutation**: `Icmp(cc, iconst, x)` currently misses
      `X86CmpI`; needs a cc-flipping rewrite.
- [ ] **Wider LEA coverage** beyond LEA2/3/4 already present; `lea` as a
      3-operand add to avoid destructive-form moves.
- [ ] **Latency/port-aware DAG scheduling.** A uarch model (ports, latencies,
      throughput) is only tractable because there is one target. ~5% but it is
      exactly the kind of win the single-target thesis predicts.
- [ ] **Shift+mask folding**: `And(Shr(a,n), mask)` when the shift already
      zeroed the masked bits (last open rule in the e-graph rule inventory).
- [ ] **MachInst peephole audit** (`src/emit/peephole.rs`): confirm `mov r,r`
      elimination, LEA shrinking, jmp-to-fallthrough removal.
- [ ] **Branch layout**: `__builtin_expect` / likely-unlikely hints and
      profile-free heuristics driving block ordering.

### P3 -- Register allocation

- [ ] Register hints: prefer RAX for returns, ABI arg regs for last-use-before-
      call operands.
- [ ] Cross-block coalescing beyond copies; same value through a block param
      should prefer one register.
- [ ] **The splitter's victim heuristic.** `split::score_victim` is
      `live_range_length / loop_penalty`, which cannot tell a long-lived value
      read twice from one read twenty times -- on
      `regalloc/array_spill_frame_corruption.c` it picks a value stored once and
      reloaded 23 times, which is most of that program's +55%. Moved here from
      `docs/refactor-roadmap.md` step 6: the fold is done and this is allocation
      policy, not a refactor. **Two directions are closed by measurement**, and a
      replacement has to beat those numbers rather than be reasoned from first
      principles: dividing by use count (the Chaitin ratio) costs 35 regressed
      codesize rows against 12 and 88 on fuzz, tried three times now; and the
      pass in front of it, `insert_early_barrier_spills`, cannot be gated on
      pressure because it runs before global liveness exists, while deleting it
      outright improves the corpus in aggregate but costs `args` seed 3 its
      compile.
- [ ] Better spill placement: split at loop boundaries, more remat shapes
      (currently leaf/free ops only).
- [ ] Per-param precoloring: today all params are skipped when the block
      contains a call; could be decided per param at each call point.

### P4 -- Unblocks only if they gate measurement

- [ ] Switch/case + dense jump-table lowering (frontend + backend).
- [ ] `>6` int / `>8` float args via stack: verify the callee read side and
      caller-side alignment.
- [ ] SysV struct-by-value passing/returning. `assign_args`
      (`src/x86/abi.rs:112`) `unreachable!()`s on non-int/float; needs
      INTEGER/SSE/MEMORY classification and hidden-pointer return.
- [ ] Error recovery: emit a diagnostic instead of panicking on internal errors.

## Known bugs

**Four wrong-value programs and twenty-seven capacity failures are open**, found
by running the generator at 200 seeds a shape instead of the 30 every gate is
pinned at. Reproducers are kept outside the repo at `~/.cache/blitz-fuzz-fails/`
and are unreduced; `gen_c.py --seed N --shape S` regenerates one until the
generator changes, which is why the files are the durable artifact and the seed
is not.

| shape | passing | wrong value | capacity |
| --- | --- | --- | --- |
| `mixed` at 200 | 195/200 | 109 | 57, 123, 135, 150 |
| `args` at 200 | 185/200 | 52, 175 | 13 seeds |
| `pressure` at 200 | 189/200 | 128, 131, 158, 165 | 7 seeds |

**Re-measure rather than trust the list.** Four entries left it without anyone
fixing them: `mixed` 137 and 196 and `args` 146 stopped miscompiling when step
6's fold removed the single-block path's RAX dividend pin, and `args` 108 was
already passing by the time it was taken up. **`mixed` 57 went the other way** --
a capacity failure the same fold introduced.

The `pressure` row is new information rather than a regression: that shape had
never been run at this width, and every one of its wrong-value programs predates
the runs that found them.

Every remaining wrong-value program is right at `-O0` and wrong at `-O1`, so the
self-consistency oracle carries all four. The one that needed the `cc` oracle was
`mixed` 92, wrong at both levels: a block took `Proj0` of a division emitted in
another block, where RAX no longer held the quotient.

The capacity failures are the two shapes the allocator names itself: *"spilling
did not reduce it, so the pressure point is one instruction whose own operands are
what is live there"*, and *"every over-budget VReg is a block parameter, which
only the splitter can route through a slot"*.

What the fixed ones are worth is the shape they kept having, which is the first
thing to check on any of these:

- **A block resolved an e-class to the wrong VReg** -- nine bugs, and the reason
  steps 1-4 of `docs/refactor-roadmap.md` exist. `BLITZ_DEBUG=regalloc` dumps the
  final assignment, and a value with several VRegs where only one has the right
  register is the signature.
- **Liveness measured against one instruction order while another is emitted.**
- **A pseudo-op's position taken for its value's position.** `Op::BlockParam` is
  a marker; every parameter of a block already holds its register before the
  block's first instruction runs.

`CLAUDE.md` and `DEBUGGING-NOTES.md` carry these with the techniques that found
them.

**The capacity failures are not closed, and 30/30 was never evidence that they
were.** Step 5's spill loop and 5c's slot routing took all three shapes to 30/30
and that reads as a fix; at 200 seeds sixteen remain. What the two steps did buy
is real -- the same corpus was 14/30 on `pressure` before them -- but the gate's
width, not the compiler, is what made the number green.

**The lesson is about the gate, not about the bugs.** A corpus pinned at the
width where it stops failing reports its own width back as a pass. `run_fuzz.sh`
takes a seed count as its first argument and 200 seeds is 67 seconds, so nothing
but habit was keeping it at 30.

**The constant-remat lever is still open and still worth taking.** The offenders
were long-lived hash-consed constants: one `Iconst(3)` serving `arr[3]`'s index
in the entry block and a `+ 3` twenty blocks later holds a register in between,
and one fuzzer function had ~100 simultaneously live needing 117 colors.
`mov reg, imm` is one instruction with no memory traffic, so the register is
never worth keeping. Two implementations were reverted, **both on the seam step 4
has since removed rather than on the policy**:

1. A splitter pre-pass rematerializing constants at cross-block `Iconst` uses,
   with copies pinned to their consuming barrier's group. Sound and ~70 lines; it
   dropped -O1 overshoots from 14-18 to 10-15 and was rejected by
   `BLITZ_VERIFY=strict` because segment points were fixed against the post-split
   schedule while coalescing moved instructions again.
   Not attempted: `StackAddr`/`GlobalAddr`, which segfault 7 lit tests on their
   own by defeating `build_mem_addr`'s folding check, and terminator uses, which
   need a copy at block end with a segment covering block exit.
2. Re-emitting `Iconst` classes per block, reusing the flags-typed mechanism. 62
   lit failures.

## Tech debt

- [ ] `docs/split-pass-plan.md` Phase 8 and Final Audit are unchecked. The audit
      requires zero hits for `coalesce_aliases`; there are 25. Decide whether it
      is now load-bearing, then finish or amend the plan.
- [ ] Clear the clippy backlog (47 warnings, cosmetic), including the
      `unused_mut` on `let mut emit` at `src/schedule/scheduler.rs:265`.
- [ ] `README.md` quotes 917 unit / 406 lit / 268 differential and omits
      forwarding, DSE and the splitter from its pipeline diagram.
- [ ] File sizes, with no evidence of harm attached to any of them:
      `compile/tests.rs` 3751 lines, `egraph/algebraic.rs` 2289,
      `compile/mod.rs` 1865. A rule set being long is fine.

## Decisions worth not relitigating

- **Real flag fusion is rejected, not deferred.** Reusing an earlier
  `X86Sub(a,b)`'s flags for a later `Icmp(cc, diff, 0)` is unsound for signed
  ordering across overflow. Eq/Ne and unsigned only.
- **Signed-ordering algebraic rewrites stay rejected** until nsw/nuw exists.
  Regression guards: `icmp_sgt_sub_zero_not_rewritten`,
  `icmp_sgt_add_const_zero_not_rewritten`.
- **x86-64 has no flag-only ADD.** SUB has CMP, AND has TEST, ADD has nothing;
  LEA computes without flags. The `Icmp(Eq/Ne, Add(a,k), 0)` -> `Icmp(Eq/Ne, a, -k)`
  rewrite is the answer for the common case. Do not revisit as an isel pattern.
- **`must_alias` is canonical e-class equality only.** "Same base" heuristics
  belong in `may_alias`. Hashconsing gives real must-alias for free.
- **LICM runs before saturation**, on the raw e-graph, so hoisted code still
  gets the full rewrite pass.
- **Forwarding runs pre-LICM; DSE runs post-forwarding.** Order is load-bearing.
- **Cost-based extraction picks the instruction form.** Isel rules add every
  legal alternative to the class and let cost decide; no manual selection logic.
