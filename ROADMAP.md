# Blitz Roadmap

## Goal

Produce the best possible x86-64 machine code. Blitz targets **one** ISA on
purpose: the optimizer can reason natively about addressing modes, flags, LEA,
multi-output instructions, and uarch behavior instead of laundering everything
through a target-neutral IR. Anything a portable backend has to generalize away
is fair game here.

The measure of success is code quality against `gcc -O2` / `clang -O2` on the
same input: fewer instructions, fewer bytes, fewer spills, better loops.

**Where that stands: `x1.29` against `gcc -O2`**, geometric mean of the
per-program instruction ratio over the 21 `live` kernels, from
`bash tests/run_codesize.sh --gap`. Worst is `call_hot` at `x2.29`.

Read `live`, not `bench`. Both are reported, but `bench`'s kernels resist
folding only by being large enough that `gcc` gives up, so that ratio is against
whatever `gcc` happened to leave behind. Every `live` kernel seeds its data from
`argc`, which no reference compiler can evaluate.

**The `clang` column is not a ranking.** It reads `x0.78` on `live`, meaning
blitz emits fewer instructions than `clang -O2` -- because `clang` vectorizes
and unrolls these loops. This counts static instructions, so a ratio under 1.0
is a program to go and read, not a win.

Correctness is a precondition, not a tradeoff against that. An aggressive
single-target optimizer has more room to be subtly wrong than a conservative
portable one, so the correctness infrastructure in P0 is part of the goal rather
than overhead against it.

## Non-goals

- **tinyc is not a product.** It exists to feed the backend real-ish input and
  to make tests readable. Frontend work happens only when a missing C feature
  blocks a codegen experiment we want to run. Never for its own sake.
  Explicitly not planned: preprocessor, typedef/enum/union, initializer lists,
  function pointers, bitfields, diagnostics quality.
- **Multi-target.** Single-ISA specialization is the whole thesis.
- **Toolchain completeness** for now: `.eh_frame`/CFI, PIC/PIE, TLS, inline asm,
  atomics, sanitizers, LTO, PGO. These are what a *shipping* compiler needs, not
  what a *good* optimizer needs.

**DWARF is wanted eventually**, which is the one item that has moved off this
list. It is not scheduled, but it constrains decisions now: `-O0` has to stay
the level that maps to source, so a pass that deletes a value someone could
inspect does not belong there. That is the line the DCE split is drawn on, and
it is the line a fast `-O0` allocator sits on the right side of -- locals in
frame slots at fixed offsets is what debug info describes.

## Start here

Ordered. Each says what it is, why it is placed there, and what would tell you
it is done. **Items 1 and 2 are closed; item 3 is the open one.** The closed
entries stay because what they measured is the reason the next attempt should
not start where they did.

### 1. ~~Finish the `-O0` allocator~~ -- closed

**It is the `-O0` default** (`regalloc::fast`) and it is correct.
`BLITZ_PASSES=-fast-regalloc` puts `-O0` back on the colouring path, and
`+fast-regalloc` puts `-O1` on the slot path; either is the comparison to reach
for when an allocation looks wrong.

**The model.** Every value gets a frame slot; before each instruction its
operands load into fresh VRegs, the instruction writes another, and that stores
back. Nothing is held across an instruction, so no live range outlives one
expansion and a handful of registers serves the whole function. Fresh VRegs are
also how it fits `assignment`, which has one entry per VReg.

**Why it exists**, now that capacity failures are closed and it is not needed to
rescue anything: `run_diff.sh`'s `-O0`-vs-`-O1` leg is blind to anything equally
wrong at both levels, and while both levels shared an allocator the component
the bug priors rank first was the one that comparison could not see.

It is also 1.41x faster at `-O0` (211.7ms to 150.5ms on a 6048-line input),
because it skips the pressure splitter. An allocator holding nothing across an
instruction has no pressure to relieve; the only shape it cannot place is one
instruction reading more operands than the machine has registers, which no split
helps.

**State, 2026-08-07.** `-O1` output is byte-identical to before the allocator
landed, at every one of 780 identity comparisons.

```
unit    1010/1010          lit      546/546
diff    335/335 matched    0 differed        (cc oracle 335/335)
corpus  16/16, nothing open
fuzz    mixed 400/400   args 400/400   pressure 400/400
```

**The bugs were one class, not a list**: an op that *names* a value the
every-value-in-a-slot model cannot hold. EFLAGS, which no store reaches; a pair
VReg, which holds neither of the halves it names; a jump's arguments, which
must not each cost a register; an ABI register, which is where a value has to be
*at a call* rather than a property it carries everywhere; and `idiv`'s operands.
A scratch also has to carry the type of the value it stands in for, or lowering
sizes the instruction 64 bits wide by default.

**It found its first bug in the `-O1` allocator immediately**, which is what it
is for: a function parameter is in its register before the function's first
instruction runs, and `Op::Param`'s shadow was not modelled the way
`Op::BlockParam`'s was, so a splitter store took RCX while it held the fourth
argument (`tests/fuzz/corpus/fixed/args-seed310.c`).

**Two models are already ruled out; do not start on either.** A whole-live-range
linear scan cannot work under one-register-per-VReg: a value live in an early
block and a late one holds a register across everything between, block
parameters cannot be spilled, and a loop-heavy function runs out with nothing it
may take -- on `bench/sieve.c` in round 0, before any spilling. Pre-spilling
every cross-block value first is worse (refusals 55 to 67), because spilling
converts a spillable value into reloads and a reload cannot be spilled in turn,
so the pressure is relabelled rather than removed.

### 2. ~~The last capacity failure~~ -- closed

`args` seed 88 compiles, and **200 seeds a shape is clean on all three for the
first time**: `mixed` 200/200, `args` 200/200, `pressure` 200/200. (At 400
seeds a shape, which is what `run_fuzz.sh` now sweeps by default, all three are
clean too.)

What closed it was not a fifth attempt at the spill loop. The splitter models
the registers a call takes away (`callee_saved_budget`) and modelled nothing at
all for a division, where the allocator clobbers RAX and RDX as the dividend.
Its own comment states the invariant that breaks -- *"must match what the
allocator uses, or the splitter measures a different graph than the one being
coloured"* -- so it reported functions as fitting that the colouring could not
place, and left the spill loop to fail on them. Giving a division the same
treatment as a call costs `+0.67%` instructions and `+0.93%` bytes over 107
changed rows, and `live` does not move at all.

**This is the shape item 1's step three is about**, arrived at from the other
end: a pre-pass relieving pressure against a different graph than the allocator
enforces. It was under-relieving here rather than over-relieving.

### 3. Give the inliner a pressure check, as LICM has

**Re-measure before starting: this was last measured 2026-08-06 and a lot of
codegen has moved since.** As of then, `-O1` emitted worse code than `-O0` on 7
of the 15 `bench` kernels; LICM was 60% of that and is now budgeted, and the
rest was inlining, which decides without looking at pressure in exactly the same
way. `BLITZ_PASSES=-inlining` took `bench` instructions 2637 to 2422 and reloads
377 to 247. The fix has the same shape as `licm::within_budget`.

Done when `-O1` beats `-O0` on all 15 kernels.

After those, the P1 list below is ordered by measured impact. **`P2` is where
the single-target thesis pays off.**

**Do not start in the register allocator, the splitter, or the block-parameter
machinery without reading `docs/internal/refactor-roadmap.md` first.** It is
finished as work and is now the record of what eleven steps measured --
including the predictions that were wrong, which is what stops the next attempt
repeating them.

## Current state (2026-08-07)

- 1010 Rust tests + 546 lit tests, all green. `cargo fmt` clean, `cargo clippy
  --all-targets` clean, zero build warnings, zero rustdoc warnings.
- `BLITZ_VERIFY=1` and `BLITZ_VERIFY=strict` green across both suites.
- `bash tests/lit/run_diff.sh`: 335 compared `-O0`-vs-`-O1` and against a
  reference compiler; no skips, no differences under gcc or clang.
- **`-O0` is on `regalloc::fast` and both levels are correct.** Everything below
  is `-O1` unless it says otherwise.
- `bash tests/fuzz/run_corpus.sh`: 16 `fixed` programs, all passing at both
  levels, and nothing open.
- Generated programs: `mixed` 400/400, `args` 400/400, `pressure` 400/400, which
  is what `run_fuzz.sh` now sweeps by default. **The width is what makes it a
  check** -- the `-O1` allocator bug in `fixed/args-seed310.c` is at seed 310 of
  `args` alone, and at the 30 seeds the gates used to be run at, all three
  shapes were green while seven programs miscompiled.
- Code quality has a baseline: `bash tests/run_codesize.sh --check`, over `lit`,
  `bench`, `live` and `fuzz`, fed by `BLITZ_DEBUG=stats`.
- Code quality also has an *absolute* number, which is the one the Goal is
  written against: `--gap`, `x1.29` vs `gcc -O2` over the 21 `live` kernels.
  `lit` and `fuzz` compute a fixed answer from no runtime input, so `gcc -O2`
  evaluates the whole program and emits the constant. `--gap` detects that where
  it is total but not where it is partial, and partial is the common case.
  `live` seeds every kernel from `argc` and cannot be folded at all. Widening it
  further has no natural ceiling and is always a valid use of leftover time.
- **Compile time is superlinear in blocks x classes**: `secs ~ (B*C)^0.86`,
  R2=0.92, and **both levels sit on one line**. `-O1` is not intrinsically
  cheaper, it hands the same pipeline a smaller IR. On a 6048-line input the two
  levels are now 246ms (`-O0`) and 250ms (`-O1`). The remaining Theta(B*C) site
  is the splitter's pressure scan; linearize's per-block class-map
  evict-and-restore has been narrowed. `bash tests/profile.sh <src> [flags]` is
  the way in; `perf report` hangs on these profiles, `perf script` does not.
- Pipeline: IR -> inlining -> DCE1 -> store/load forwarding -> DSE -> LICM ->
  e-graph saturation -> cost-based extraction -> DCE2 -> linearize -> trivial
  block-parameter removal (re-extract + linearize again) -> DAG schedule ->
  live-range splitter (`-O1`, and only where the colouring fails) -> regalloc
  (`-O1` function-scope Chaitin-Briggs, `-O0` every value in a frame slot) ->
  terminator lowering -> MachInst lowering -> branch relaxation -> ELF.
- Implemented e-graph rules: see `docs/internal/egraph-optimization-roadmap.md`.
- Splitter design: see `docs/internal/split-pass-plan.md`.

## Priorities

### P0 -- Correctness

A fast wrong answer is worthless, and an optimizer this size certainly has
subtle bugs left. Every miscompile of the last several months was found by the
generated corpus and its two oracles, not by a hand-written test happening to
hit the right shape -- which is the argument for keeping the generator ahead of
the compiler rather than treating a green suite as evidence. The priors on where
bugs live: **regalloc** (splitting, coalescing, spill/remat, cross-block
liveness) first by a wide margin, then the **memory passes** (forwarding/DSE
resting on a conservative alias model), then **isel width and type handling**.

**What exists**, all green and described in `CLAUDE.md`: the `-O0`-vs-`-O1`
differential with a `cc` oracle beside it (`run_diff.sh`), the UB-free generator
and its shrinker (`gen_c.py`, `reduce.py`), the per-pass IR verifier and its
strict mode (`BLITZ_VERIFY`), and the machine-level verifier over the final
instruction stream. The gate set is **fixed at four runs**; new invariants go
inside the runs that already happen. A battery that grows every time something
is learned stops being run between every change, and one-change-at-a-time is
what makes attribution possible here.

**Diagnostics worth building, each earned by a session it would have shortened.**
None is a gate; they are what turns a wall of numbers into a name. Ordered by
hours lost, not by effort to build.

- [ ] **Make the per-function VReg numbering impossible to miss.** Numbering
      restarts at v0 in every function and the dumps repeat bare `v14` on every
      line, so any analysis over a whole `--emit` run silently pools `v5` in
      `main` with `v5` in `f0`. That manufactured a measured "5 flags values live
      at once" that did not exist and nearly sent a refactor the wrong way. Print
      `f0:v14`, or state the scoping in the header of every block.
- [ ] **A `BLITZ_DEBUG=spill` category.** Spilling is the only major pass with no
      trace of its own: what it chose, whether the choice was slot or remat, and
      what the slot was. Today the choice is inferred from `slots committed=0`
      meaning "it must have rematerialized". Note the trap it has to avoid: read
      the choice off the list the *colouring* ran on, never off the result -- a
      rematerialized VReg's defining instruction is already dropped there, so
      every candidate reads as "no def" and points at a bug that is not there.
- [ ] **Name the pass that produced each number in a `--check` regression.** A
      row moving is currently attributed by re-running with `BLITZ_PASSES=-x`
      one flag at a time. The stats already exist per function; recording which
      passes ran beside them would make the bisection a lookup.
- [ ] **A decision diff, not an output diff.** `run_identity.sh` says the
      emitted code changed; nothing says *which decision* changed. Two runs'
      allocation, coalescing and hoisting choices, diffed by VReg, would
      attribute a regression in one step instead of by bisection.
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
      Cheap, and the only systematic defense against a rule that is right for
      most inputs. The two rejected signed-ordering rewrites are the cautionary
      example.
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

**What exists**: `bash tests/run_codesize.sh [--check|--update|--gap]` over four
corpora with a baseline each in `tests/baselines/`, fed by `BLITZ_DEBUG=stats`.
Instruction count, `.text` bytes, spill stores and reloads per (program, level);
a generated program that does not compile is a `-` row rather than an omission,
so the holes stay visible.

- [ ] Wall-clock via hyperfine, and the same table for `gcc -O2` / `clang -O2`
      beside blitz's own numbers. Deferred deliberately: blitz-against-itself is
      what refereed the decisions so far, and an external column moves when the
      system compiler updates.

### P1 -- Optimizer gaps with the largest measured impact

- [ ] **Copies are still a third of what blitz emits.** Measured over the 15
      `bench` kernels: blitz 805 register-to-register moves in 2613
      instructions, against `gcc -O2`'s 345 in 2226 and `clang -O2`'s 214 in
      2572. **The whole remaining instruction gap to gcc is copies.**
      Conservative coalescing is at its limit: with Briggs and George both in,
      34 of 64 candidate copies on `queens` and 43 of 112 on `hash_table` are
      still refused because the merge genuinely constrains the graph
      (`BLITZ_DEBUG=coalesce` reports the declines). Getting further needs a
      structural change, and **iterated coalescing is measured out** -- see
      Decisions. *Fewer block parameters to copy* is the candidate left, and
      `docs/internal/refactor-roadmap.md` argues it at length. Note
      `phi_removal` already does both tiers including self-references, so the
      82% of parameters `count_trivial_phis.py` calls redundant on `hash_table`
      is what the *rule* permits, not what is sound to remove -- one e-class is
      one expression, not one value. Read that file before starting.
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

- [ ] **A cost model that can see a constant being materialized.** This blocks
      three separate items, which is why it leads the list. `Iconst` costs 0.0,
      so the `mov r, imm32` a register form needs is invisible to extraction and
      any immediate form has to carry the credit itself. Consequences today: the
      `imm32` ALU form is not selected (only `imm8`), worth a measured -1.4pp on
      `lit` and `fuzz` and -0.13pp on `bench`; the demanded-bits direction of
      shift+mask folding cannot be done; and store-of-immediate and the
      3-operand LEA had to be peepholes rather than isel rules. The credit is
      owed only where the constant has a single use, and **the e-graph has no
      parents map to ask**. That is the actual missing structure. If it lands,
      re-check `tests/lit/control/main_falls_off_end.c`, which is what pricing
      the `imm32` form to win broke last time.
- [ ] **Bit instructions**: `popcnt`, `bsr`/`bsf`, `tzcnt`/`lzcnt`, `bswap`, and
      `bt` itself -- the read form, whose result is a flag and so needs the
      compare seam rather than a value class. `bt` needs a cc-carrying node like
      `X86UcomisdCc` so an `Icmp` class can take its flags from a CF-only
      instruction with the cc rewritten `Ne -> Ult`.
      **The blocker on the rest is a source idiom, not the encoding**: tinyc has
      no builtins, so nothing reaches these rules. `bswap` (a 4-way `Or` of
      masked shifts) is the only plausible one to match today.
      `bts`/`btr`/`btc` are done in both register and immediate-index forms.
- [ ] **BMI/BMI2 when available**: `andn`, `bextr`, `blsi`/`blsr`/`blsmsk`,
      `shlx`/`shrx`/`sarx` (no flag clobber, no CL constraint), `mulx`.
      Needs a CPU feature level knob that does not exist yet.
- [ ] **Carry-chain `adc`/`sbb` proper.** A multi-word add has no source idiom
      in tinyc, so this needs a shape to match (`a + b`, then `c + (sum < a)`)
      or nothing reaches it. The `setcc`-free 0/-1 mask is done, `Ult` only:
      `Ugt` and `Ule` would need the compare's operands swapped, which is a
      different compare, and the signed conditions are not the carry flag at all.
- [ ] **Wider LEA coverage** beyond LEA2/3/4 and the 3-operand add already
      present.
- [ ] **Latency/port-aware DAG scheduling.** A uarch model (ports, latencies,
      throughput) is only tractable because there is one target. ~5% but it is
      exactly the kind of win the single-target thesis predicts.
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
      reloaded 23 times, which is most of that program's +55%. **Two directions
      are closed by measurement** and a replacement has to beat those numbers
      rather than be reasoned from first principles: the Chaitin ratio (dividing
      by use count) costs 35 regressed codesize rows against 12, tried three
      times; and `insert_early_barrier_spills` cannot be gated on pressure
      because it runs before global liveness exists, while deleting it outright
      improves the corpus in aggregate but costs `args` seed 3 its compile.
- [ ] Better spill placement: split at loop boundaries, more remat shapes
      (currently leaf/free ops only).
- [ ] Per-param precoloring: today all params are skipped when the block
      contains a call; could be decided per param at each call point.
- [ ] **The constant-remat lever is still open and still worth taking.** The
      offenders are long-lived hash-consed constants: one `Iconst(3)` serving
      `arr[3]`'s index in the entry block and a `+ 3` twenty blocks later holds a
      register in between, and one fuzzer function had ~100 simultaneously live
      needing 117 colors. `mov reg, imm` is one instruction with no memory
      traffic, so the register is never worth keeping. Two implementations were
      reverted, **both on the seam refactor step 4 has since removed rather than
      on the policy**: a splitter pre-pass rematerializing at cross-block
      `Iconst` uses (sound, ~70 lines, dropped `-O1` overshoots from 14-18 to
      10-15, rejected by `BLITZ_VERIFY=strict` because segment points were fixed
      against the post-split schedule while coalescing moved instructions
      again), and re-emitting `Iconst` classes per block (62 lit failures).
      Not attempted: `StackAddr`/`GlobalAddr`, which segfault 7 lit tests on
      their own by defeating `build_mem_addr`'s folding check, and terminator
      uses, which need a copy at block end with a segment covering block exit.

### P4 -- Unblocks only if they gate measurement

- [ ] Switch/case + dense jump-table lowering (frontend + backend).
- [ ] `>6` int / `>8` float args via stack: verify the callee read side and
      caller-side alignment.
- [ ] SysV struct-by-value passing/returning. `assign_args`
      (`src/x86/abi.rs:112`) `unreachable!()`s on non-int/float; needs
      INTEGER/SSE/MEMORY classification and hidden-pointer return.
- [ ] Error recovery: emit a diagnostic instead of panicking on internal errors.

## Known bugs

**Nothing is open.** No wrong-value programs and no capacity failures at either
level: at 400 seeds a shape `mixed`, `args` and `pressure` are all 400/400, and
the saved corpus is 16 passing with an empty `open/`. The last one, an `-O1`
allocation bug the `-O0` allocator's arrival made visible, closed when
`Op::Param` got the shadow `Op::BlockParam` already had -- see item 1 of Start
here.

**Re-measure rather than trust that.** Entries have left this list without
anyone fixing them, and one went the other way -- a capacity failure that a fold
introduced. The files under `tests/fuzz/corpus/` are the durable artifact and
the seed is not, since `gen_c.py --seed N --shape S` only regenerates a program
until the generator changes.

**What the fixed ones are worth is the shape they kept having**, which is the
first thing to check on any wrong-value bug:

- **A block resolved an e-class to the wrong VReg** -- nine bugs, and the reason
  steps 1-4 of `docs/internal/refactor-roadmap.md` exist. `BLITZ_DEBUG=regalloc`
  dumps the final assignment, and a value with several VRegs where only one has
  the right register is the signature. `BLITZ_DEBUG=paramsrc` prints the
  block-parameter form of the disagreement directly.
- **Liveness measured against one instruction order while another is emitted.**
- **A pseudo-op's position taken for its value's position.** `Op::BlockParam`
  and `Op::Param` are markers; every parameter of a block already holds its
  register before the block's first instruction runs, whether the phi copies on
  the edge wrote it or the caller did. `Op::is_param_marker` is the one place
  that says which ops these are.

`CLAUDE.md` and `DEBUGGING-NOTES.md` carry these with the techniques that found
them.

**A corpus pinned at the width where it stops failing reports its own width back
as a pass.** All three shapes read 30/30 while seven programs miscompiled, and
the `-O1` allocation bug in `fixed/args-seed310.c` needed seed 310 of `args`.
`run_fuzz.sh` now defaults to 400 seeds of all three shapes, which is 164
seconds -- what the other harnesses already cost.

## Tech debt

- [ ] `docs/internal/split-pass-plan.md` Phase 8 and Final Audit are unchecked.
      The audit requires zero hits for `coalesce_aliases`; there are 25. Decide
      whether it is now load-bearing, then finish or amend the plan.
- [ ] File sizes, with no evidence of harm attached to any of them:
      `compile/tests.rs` 3751 lines, `egraph/algebraic.rs` 2289,
      `compile/mod.rs` 1865. A rule set being long is fine.

## Decisions worth not relitigating

- **`-O0` is not slower than `-O1`, and the reason it used to be is settled.**
  Compile time is `~(B*C)^0.86` with both levels on one curve, so `-O1` was
  never intrinsically cheaper -- it handed the same pipeline a smaller IR. DCE
  was the whole of it: on a 6048-line input it deletes 2519 of 3763 blocks and
  zero dead loads. The CFG half of DCE now runs at every level and the dead-load
  half stays on `-O1`. Do not re-open this as a profiling question.
- **Unreachable-block elimination is canonicalization, not optimization.** A
  block no path reaches costs the scheduler, splitter and allocator their full
  price for code that cannot run. Removing a dead *load* is the optimization,
  because it takes away a read of a value someone may want to inspect -- which
  is the same line gcc and clang draw at `-O0` for debug info.
- **Iterated coalescing is worthless here.** George & Appel's leverage is
  `simplify`, which removes every node of degree < k so the rest fall below the
  threshold the two tests are stated against. Simulated on the `bench` kernels
  it removes 62 of 165 nodes on `queens` and 80 of 297 on `hash_table`, and
  **zero** refused copies would pass afterwards: the survivors have their
  endpoints in the dense core, which simplification does not touch. Do not write
  the worklist allocator for this.
- **Offset-aware alias analysis is in and its measured effect is close to
  nothing** -- one `lit` row moves and `struct_walk` goes the other way by 2.7%
  because better forwarding keeps more values live. The capability is real and
  `tests/lit/alias/forward_across_struct_field.c` covers it; the corpus cannot
  price it, because `gen_c.py` does not generate structs at all. That is a gap
  in the corpus, not a reason to revisit the pass.
- **Real flag fusion is rejected, not deferred.** Reusing an earlier
  `X86Sub(a,b)`'s flags for a later `Icmp(cc, diff, 0)` is unsound for signed
  ordering across overflow. Eq/Ne and unsigned only.
- **Signed-ordering algebraic rewrites stay rejected** until nsw/nuw exists.
  Regression guards: `icmp_sgt_sub_zero_not_rewritten`,
  `icmp_sgt_add_const_zero_not_rewritten`.
- **x86-64 has no flag-only ADD.** SUB has CMP, AND has TEST, ADD has nothing;
  LEA computes without flags. The `Icmp(Eq/Ne, Add(a,k), 0)` -> `Icmp(Eq/Ne, a,
  -k)` rewrite is the answer for the common case. Do not revisit as an isel
  pattern.
- **`must_alias` is canonical e-class equality only.** "Same base" heuristics
  belong in `may_alias`. Hashconsing gives real must-alias for free.
- **LICM runs before saturation**, on the raw e-graph, so hoisted code still
  gets the full rewrite pass.
- **Forwarding runs pre-LICM; DSE runs post-forwarding.** Order is load-bearing.
- **Cost-based extraction picks the instruction form.** Isel rules add every
  legal alternative to the class and let cost decide; no manual selection logic.
- **There is no `ror` or `shrd` isel form.** `ror k` is `rol (w-k)` in the same
  encoding size and `shrd y,x,w-k` computes the same bits as `shld x,y,k`, so
  the pairs differ only in which operand the destructive form consumes and
  extraction has no basis to prefer either.
