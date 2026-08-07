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
it is done.

### 1. Give `-O0` a fast allocator instead of the colouring one

`-O0` and `-O1` run the **same** pressure splitter and the same function-scope
Chaitin-Briggs allocator. That is the problem, and compile time is the least of
it.

**The correctness argument is the real one.** The bug priors in P0 put regalloc
first by a wide margin, and `run_diff.sh`'s `-O0`-vs-`-O1` oracle is by
construction blind to anything equally wrong at both levels. Sharing the
allocator means the highest-risk component in the compiler is exactly the one
the primary oracle cannot see; every allocator bug so far was caught by the `cc`
reference leg or by reading asm. A separate `-O0` allocator makes allocator bugs
visible to self-consistency for the first time.

Two more payoffs. **Capacity failures stop existing at `-O0`**: a fast allocator
spills rather than refusing to colour, so every generated program compiles at
one level and gets judged -- see item 2, which is the last one left, and the
note there about unjudged programs. And **it is what DWARF will want**: locals
in frame slots at fixed offsets is what `-O0` debug info describes, where the
colouring allocator's whole job is to keep values in registers.

**A whole-live-range linear scan was tried and is measured out. Do not start
there again.** `src/regalloc/fast.rs` is that attempt, behind
`BLITZ_PASSES=+fast-regalloc`. Forced on it reaches 268 of 334 differential
comparisons and refuses 55 programs, and the refusals are structural rather than
a list of bugs: keeping `allocate_global`'s interface means one physical register
per VReg for the whole function, so a value live in an early block and a late one
holds a register across everything between. Blocks laid end to end give no holes
to exploit. Pressure is then high everywhere, which would only mean heavy
spilling -- except that block parameters cannot be spilled at all, since a phi
copy on the edge writes them. A loop-heavy function has many long-lived
parameters live at once and the scan runs out with nothing it may take. On
`tests/lit/bench/sieve.c` this happens in round 0, before any spilling: 48
intervals, and `v19` needs a register no value can give up.

**Pre-spilling every cross-block value before the scan was tried and is also
measured out**, which is the same finding from the other side. It is the obvious
repair -- take the long-lived values out first so only block-local ones and the
parameters compete -- and it makes things worse: refusals 55 to 67, on `sieve.c`
the same `v19` is unplaceable with ten values already in slots. The reason is
that spilling converts a long-lived *spillable* value into reloads, and a reload
is unspillable in turn, so the pressure is not removed but relabelled as
pressure nothing may relieve. `insert_spills_global` places a reload per
(value, block) rather than per use, so those reloads are not short either.

The lesson is that the whole-range model and the one-register-per-VReg interface
cannot both hold. **The per-instruction scratch model does not have this failure
mode**, because nothing is held across an instruction: every interval is about
one instruction long, so long live ranges stop mattering and many VRegs share a
register without ever being live together. It reaches `vreg_to_reg` by rewriting
the stream so each use loads into a *fresh* short-lived VReg and each definition
stores back -- the same interface, different VRegs.

**And the pipeline, not the allocator, is where this is decided.**
`docs/internal/refactor-roadmap.md` measured it while folding the two allocators
into one: *"Skip the splitter for single-block functions and let the global
allocator's spill loop cope: 159 of 474 lit tests fail, most of them compilation
failures. The spill loop is not a substitute for the splitter even on one
block."* And it names the cause: *"What blocks step 6 is not the merge; it is
that the passes in front of the allocator spill before knowing whether the
allocator needs them to."* A second allocator with its own pressure model
inherits relief planned against the colouring allocator's, which is why the
attempt above failed with the splitter running. Read that file first.

So the order is: teach the allocator result to say where a value lives, then
make pressure relief something the allocator asks for rather than a pre-pass
guesses at, and only then write the second allocator.

**Step one is in** (`1a4b1cb`): `Assignment` is `Reg(Reg) | Slot(u32)` and the
result field is `assignment`. The colouring allocator constructs only
`Assignment::Reg`, so it changed no emitted code on any corpus.

**Step two, started, with the target now named.** The goal is that a value in a
slot is a value like any other, so `terminator.rs`'s two silent `continue`s --
reached when a VReg has neither a register nor a `BlockParamSlotMap` entry, on a
path whose failure mode is a non-terminating loop -- have something to do rather
than something to skip.

**They are not dead code, and that is measured.** Turning both into
`CompileError` fails two saved corpus programs at `-O1`, both on the destination
branch:

```
fixed/pressure-seed128.c   b9  -> 130 p0   VReg(10)  class 673
fixed/pressure-seed35.c    b69 -> 149 p0   VReg(13)  class 1111
```

So a block parameter with no register and no side-table entry is a real state
and the skip is load-bearing: the value reaches the block by some route neither
map records. **Those two programs are the reproducers, and finding what route
that is is the next step** -- once it can be named as `Assignment::Slot`, the
branch does the store and a remaining `None` becomes the error it should be.

(An `eprintln!` probe in both branches printed nothing across every corpus and
was simply wrong: `run_corpus.sh` redirects the compiler's stderr. Rule out the
scaffolding first.)

Note what does *not* move: `SlotSpilledParamInfo::value_alias` says whether a
parameter names the value it carries, which decides whether a back edge must
store. That is edge identity, not storage, and
`docs/internal/refactor-roadmap.md`'s section "A back edge is not 'the VRegs are
equal'" is why it needs care.

The shape to aim for is every VReg in a slot, operands loaded into scratch
registers per instruction, results stored back: no interference graph, no
coalescing, no splitting, linear in instructions. Derive the details from
blitz's own constraints rather than from another compiler's fast allocator --
the reason there is one here is oracle independence, where LLVM's is compile
time at scale, and the two do not want the same design. What it still has to
honour, none of it optional:

- precoloring and the SysV ABI at calls (`compile/precolor.rs`), including AL on
  every call
- block parameters, which are written by phi copies on the edge before the
  block's first instruction runs
- `regalloc::SlotAllocator`, which owns one frame-slot numbering per function and
  records the pass each slot belongs to -- a second allocator is a fourth
  spilling pass, not an exception to that rule
- the machine-level verifier (`BLITZ_VERIFY`): no VReg surviving allocation, no
  physical register read unwritten on some path

**This reverses "the only allocator"**, which `CLAUDE.md` records as a
deliberate consolidation. Reversing it is the point rather than an oversight,
but say so in the commit.

Expect `-O0` code quality to drop a lot and every `-O0` codesize baseline to
churn. That is fine and it is not a regression: `-O0` quality was never a goal.
It does mean `-O0` rows stop being a quality signal, so read `-O1` rows after
this lands.

Done when: `-O0` uses the fast path, all four corpora and the fuzz shapes are
green at both levels, `args` seed 88 compiles at `-O0`, and `-O1` codesize rows
are **unchanged** -- the change must not reach the optimized level at all.

### 2. The last capacity failure

`tests/fuzz/corpus/open/args-seed88.c`, and at 200 seeds a shape it is the only
one: `mixed` 200/200, `args` 199/200, `pressure` 200/200. Every other failure in
the table this file used to carry has closed.

The allocator names the shape itself: *"nothing live at the pressure point can
be spilled: every value there is read by the instruction, a block parameter, or
a result the hardware pins to a register"*. Thirteen block parameters feed a
`TerminatorArgs` that reads all thirteen as its own operands, so spilling cannot
relieve it -- the values are live at that point *because the instruction there
reads them*. Only the splitter can, by routing a parameter through a slot before
the schedules are built.

**A program that does not compile is a program no oracle can judge**, which is
the second reason to close it rather than a cosmetic one. Item 1 makes this
program compile at `-O0` and so judgeable, but does not fix it at `-O1`; the two
are worth doing in that order for exactly that reason.

Before trying a fifth approach, read the four measured ones at the end of
`docs/internal/refactor-roadmap.md`. Done when a 200-seed run of each shape is
clean at both levels.

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

- 1010 Rust tests + 543 lit tests, all green. `cargo fmt` clean, `cargo clippy
  --all-targets` clean, zero build warnings, zero rustdoc warnings.
- `BLITZ_VERIFY=1` and `BLITZ_VERIFY=strict` green across both suites.
- `bash tests/lit/run_diff.sh`: 334 compared `-O0`-vs-`-O1` and against a
  reference compiler; no skips, no differences under gcc or clang.
- `bash tests/fuzz/run_corpus.sh`: 13 `fixed` pass, 1 `open` fails as recorded.
- Generated programs at 200 seeds a shape: `mixed` 200/200, `args` 199/200,
  `pressure` 200/200. **No wrong-value programs and one capacity failure.**
  At the 30 seeds every gate is pinned to, all three are clean -- **that width
  measures nothing**, and `run_corpus.sh` is what compensates for it.
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
  live-range splitter -> function-scope Chaitin-Briggs regalloc -> terminator
  lowering -> MachInst lowering -> branch relaxation -> ELF.
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

**No wrong-value programs are open.** One capacity failure is: `args` seed 88,
checked in at `tests/fuzz/corpus/open/args-seed88.c`, described in item 2 of
Start here. At 200 seeds a shape `mixed` and `pressure` are clean and `args` is
199/200.

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
- **A pseudo-op's position taken for its value's position.** `Op::BlockParam` is
  a marker; every parameter of a block already holds its register before the
  block's first instruction runs.

`CLAUDE.md` and `DEBUGGING-NOTES.md` carry these with the techniques that found
them.

**A corpus pinned at the width where it stops failing reports its own width back
as a pass.** All three shapes read 30/30 while seven programs miscompiled.
`run_fuzz.sh` takes a seed count as its first argument and 200 seeds is ~67
seconds a shape, so nothing but habit keeps the gate at 30 -- which is what
`run_corpus.sh` exists to compensate for.

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
