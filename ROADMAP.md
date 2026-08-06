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

**Where that stands: `x1.36` against `gcc -O2`**, geometric mean of the
per-program instruction ratio over the 8 `live` kernels, from
`bash tests/run_codesize.sh --gap`. Worst is `call_hot` at `x2.44`.

Read `live`, not `bench`. Both are reported, and `bench` says `x1.29` -- but its
kernels resist folding only by being large enough that `gcc` gives up, so that
ratio is against whatever `gcc` happened to leave behind. Every `live` kernel
seeds its data from `argc`, which no reference compiler can evaluate.

**The `clang` column is not a ranking.** It reads `x0.74` on `live`, meaning
blitz emits fewer instructions than `clang -O2` -- because `clang` vectorizes
and unrolls these loops into 1315 instructions against blitz's 831, and its code
is much faster. This counts static instructions, so a ratio under 1.0 is a
program to go and read, not a win.

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

1. **Sixteen capacity failures are what is left.** No generated program
   computes a wrong value any more: 600 at 200 seeds a shape are `mixed`
   198/200, `args` 188/200, `pressure` 198/200, and **every one of those
   failures is the allocator refusing to colour, not a wrong answer.**
   Reproducers in `tests/fuzz/corpus/open/`; `run_corpus.sh` checks them in
   seconds.

   **A program that does not compile is a program no oracle can judge**: 21 of
   the 1200 (program, level) pairs are unjudged for that reason, so "no wrong
   value remains" is a statement about the 1179 that ran, not about all 1200.
   That is the second reason to close these. The count moves with the schedule
   in both directions: immediate-form ALU cost `args` two programs, and dropping
   the def's interference with the operand it overwrites returned eight.

   The allocator names the shape itself: *"spilling did not reduce it, so the
   pressure point is one instruction whose own operands are what is live
   there"*. Spilling cannot relieve a value that is live at a point *because
   the instruction there reads it*, which is why the spill loop stops -- see
   the four measured attempts at the end of `docs/internal/refactor-roadmap.md` before
   trying a fifth. `args` is where it concentrates, 12 of the 16. Done when a
   200-seed run of each shape is clean.
2. ~~**Make the gate able to see them.**~~ Done: `tests/fuzz/corpus/` plus
   `run_corpus.sh`, and `oracles.sh` so the saved programs and the generated ones
   are judged by the same three oracles. Save a failing program there before
   chasing it -- `run_fuzz.sh` is still pinned at 30 seeds everywhere, and 30
   seeds is where seven miscompiles hid behind three green shapes.
3. **Give the inliner a pressure check, as LICM now has.** `-O1` still emits
   worse code than `-O0` on 7 of the 15 `bench` kernels. LICM was 60% of that and
   is now budgeted; the rest is inlining, which decides without looking at
   pressure in exactly the same way -- `BLITZ_PASSES=-inlining` takes `bench`
   instructions 2637 to 2422 and reloads 377 to 247. The fix has the same shape
   as `licm::within_budget`. Done when `-O1` beats `-O0` on all 15 kernels.
4. ~~**Offset-aware alias analysis.**~~ Done: `alias.rs` splits an address into
   a base expression plus a constant displacement, and two accesses off one base
   whose `[offset, offset + width)` ranges do not meet cannot clobber each
   other. **Its measured effect is close to nothing, and that is the finding**:
   one `lit` row moves (-13.4% instructions, -24.5% bytes) and `struct_walk`
   goes the other way by 2.7% because better forwarding keeps more values live.
   Nothing else in `lit`, `bench` or `fuzz` changes, because almost nothing in
   them accesses two fields of a struct -- `gen_c.py` does not generate structs
   at all. The capability is real and `tests/lit/alias/forward_across_struct_field.c`
   covers it; the corpus cannot price it. That is item 1 of P1 restated.

After those, the P1 list below is ordered by measured impact. **`P2` is where the
single-target thesis pays off**, and its first item is now in: immediate-form ALU
took the `gcc -O2` gap x1.39 -> x1.34 on its own. The rest of P2 is untouched.

**Do not start in the register allocator, the splitter, or the block-parameter
machinery without reading `docs/internal/refactor-roadmap.md` first.** It is finished as
work and is now the record of what eleven steps measured -- including the
predictions that were wrong, which is what stops the next attempt repeating them.
Three are worth knowing before touching the splitter: the Chaitin ratio has been
rejected by measurement three times, `insert_early_barrier_spills` cannot be
gated on pressure because it runs before global liveness exists, and
`Op::BlockParam` cannot leave the e-graph because expressions consume it.

## Current state (2026-08-06)

- 925 Rust tests + 480 lit tests, all green. `cargo fmt` clean, zero build warnings.
- `BLITZ_VERIFY=1` and `BLITZ_VERIFY=strict` green across both suites.
- `bash tests/lit/run_diff.sh`: 302 compared O0-vs-O1 and against a reference
  compiler; no skips, no differences under gcc or clang.
- `bash tests/fuzz/run_corpus.sh`: the saved corpus, seconds. 8 `fixed` pass,
  2 `open` fail as recorded -- both capacity, neither a wrong value.
- Generated programs at 30 seeds a shape -- the width every gate runs -- are
  `mixed` 30/30, `args` 30/30, `pressure` 30/30. **That width measures nothing**,
  and it is what `run_corpus.sh` exists to compensate for. At 200 seeds it is
  `mixed` 198/200, `args` 188/200, `pressure` 198/200: **no wrong-value programs
  and 16 capacity failures.**
- Code quality has a baseline: `bash tests/run_codesize.sh --check`, 894 rows
  across `lit`, `bench` and `fuzz`. **`-O1` emits worse code than `-O0` on 7 of
  the 15 `bench` kernels**, and LICM is 60% of it -- see P1 below.
- Code quality also has an *absolute* number, which is the one the Goal is
  written against: `--gap`, `x1.36` vs `gcc -O2` over the 8 `live` kernels.
  `lit` and `fuzz` compute a fixed answer from no runtime input, so `gcc -O2`
  evaluates the whole program and emits the constant -- a generated program
  becomes `mov $0x562,%esi; call printf`. `--gap` detects that where it is
  total but not where it is partial, and partial is the common case. `bench`
  resists only by being big and reports a flattering `x1.29`; `live` seeds every
  kernel from `argc` and cannot be folded at all. **8 kernels is still a thin
  basis** -- widening `live` is the cheapest way to make every later quality
  claim mean more.
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
- Implemented e-graph rules: see `docs/internal/egraph-optimization-roadmap.md`.
- Splitter design: see `docs/internal/split-pass-plan.md`.

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

**Diagnostics worth building, each earned by a session it would have shortened.**
None of these is a gate; they are what turns a wall of numbers into a name. The
ordering is by hours lost, not by effort to build.

- [ ] **Put the over-budget histogram in the error, not just the trace.** The
      allocator's failure names a count -- "over-budget VRegs=48, of which
      spillable=21" -- and a peak. It does not say *what* those values are, and
      the answer settles the whole question: 21 of 24 on `args` seed 61 are
      `Pure(BlockParam)`, which says "this is the block-parameter wall" rather
      than "spilling failed". `BLITZ_DEBUG=regalloc` prints it now; the error
      itself should carry the top three defining ops with counts. Cost of not
      having it: most of a session spent on the spill loop, which was not the
      cause.
- [ ] **Say which register class each VReg is in, in the liveness dump.** There
      was no way to see that a value lived in EFLAGS rather than a GPR short of
      joining the `sched` and `liveness` dumps and re-deriving it from the op
      defining each operand. One column would have shown 35 of 196 values in the
      wrong class immediately.
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

- [x] **A benchmark corpus whose inputs are not known at compile time.**
      `tests/lit/live`, 8 kernels seeding their data from `argc`: a strided array
      walk, a struct-field walk, a dependent pointer chase, two float
      reductions, a hot loop over a `noinline` callee, a dense matmul, a
      data-dependent branch filter, and a bit-manipulation loop. `--gap` now
      defaults to `bench live`, and `live` is the one to read.
      **It disagreed with `bench` immediately: `x1.36` against `gcc -O2` where
      `bench` says `x1.29`.** `bench` resists folding only by being large enough
      that `gcc` gives up, so its ratio is against whatever `gcc` left behind;
      `live` cannot be folded at all. They are lit tests too, so `run_tests.sh`
      and `run_diff.sh` check each still computes the right answer at both
      levels and against `cc`. Widening this from 8 is the cheapest way to make
      every later quality claim mean more.
- [x] **LICM has a pressure check.** Hoisting is budgeted by the register file
      less what the loop already needs (`licm::within_budget`). instructions
      -12.3% on `bench` and -21.3% on `fuzz`, spills -91.1% and -47.3%, no row
      worse on any corpus, `gcc -O2` gap x1.42 -> x1.39. **The same defect is
      still open in the inliner** -- see item 3 of Start here.
- [x] **Offset-aware alias analysis.** `alias.rs::split_offset` takes an address
      apart into a base expression and a constant displacement; two accesses off
      one base are disjoint when their byte ranges do not meet. `s->a` and
      `s->b` stop killing each other. The corpora barely exercise it -- see item
      4 of Start here, and the corpus item at the top of this list.
- [x] **The def no longer interferes with the operand it overwrites.** x86
      arithmetic is two-address, and `build_interference_into` gave every result
      an edge to `live_at[i]` -- the set live *before* the instruction, which
      contains that operand. The form was unsatisfiable by construction and
      every such op cost a `mov`. `gcc -O2` gap x1.34 -> x1.29, `clang -O2`
      x1.07 -> x1.04; over changed rows `lit` -5.5% insts, `fuzz` -3.7%, `bench`
      -3.6%. It also removed **eight capacity failures**, 24 -> 16: the spurious
      edges were making graphs uncolourable that are not.
- [ ] **Copies are still a third of what blitz emits.** Measured over the 15
      `bench` kernels: blitz 805 register-to-register moves in 2613
      instructions, against `gcc -O2`'s 345 in 2226 and `clang -O2`'s 214 in
      2572. **The whole remaining instruction gap to gcc is copies.**
      Conservative coalescing is at its limit: `BLITZ_DEBUG=coalesce` now
      reports the declines, and with Briggs and George both in, 34 of 64
      candidate copies on `queens` and 43 of 112 on `hash_table` are still
      refused because the merge genuinely constrains the graph. Getting further
      needs a structural change rather than more tuning, and **one of the two
      candidates is now measured out**:

      - *Iterated coalescing is worthless here.* George & Appel's leverage is
        `simplify`, which removes every node of degree < k so the rest fall
        below the threshold the two tests are stated against. Simulated on the
        `bench` kernels it removes 62 of 165 nodes on `queens` and 80 of 297 on
        `hash_table`, and **zero** refused copies would pass afterwards: the
        survivors have their endpoints in the dense core, which is what
        simplification does not touch. Re-testing to a fixpoint, which is what
        landed, is worth -0.2% on `fuzz`. Do not write the worklist allocator
        for this.
      - *Fewer block parameters to copy* is the candidate left, and
        `docs/internal/refactor-roadmap.md` argues it at length. Note `phi_removal`
        already does both tiers including self-references, so the 82% of
        parameters `count_trivial_phis.py` calls redundant on `hash_table` is
        what the *rule* permits, not what is sound to remove -- one e-class is
        one expression, not one value. Read that file before starting.
- [x] **Coalescing takes George's rule as well as Briggs'.** Either test
      passing admits the merge; both are conservative, so the pair is too. Over
      the rows that changed: `fuzz` -7.1% insts and -4.5% bytes, `lit` -5.6% and
      -3.9%, `bench` -1.0%. It helps where the interference graph is dense,
      which is where Briggs alone refuses most merges.
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

- [x] **Immediate-form ALU.** `X86AddI`/`X86SubI`/`X86AndI`/`X86OrI`/`X86XorI`,
      one child producing `Pair(childtype, Flags)` exactly as the register form
      does. `gcc -O2` gap x1.39 -> x1.34, `clang -O2` x1.11 -> x1.07, worst
      kernel x2.82 -> x2.70; over the rows that changed, `bench` -3.8% insts and
      -5.1% bytes, `lit` -2.3%/-2.7%, `fuzz` -1.7%/-2.2%.
      **Only the `imm8` form is selected.** `Iconst` costs 0.0, so the `mov r,
      imm32` the register form needs is invisible to extraction and the
      immediate form has to carry the credit itself; an `imm8` form is 3 bytes
      against that form's 7 and wins outright, while an `imm32` form is 6
      against 7 and pricing it to win costs
      `tests/lit/control/main_falls_off_end.c` its compile. Measured, the wide
      case is worth a further -1.4pp on `lit` and `fuzz` and -0.13pp on `bench`.
      It comes back when the credit can be made honest -- it is owed only where
      the constant has a single use, and the e-graph has no parents map to ask.
- [ ] **Bit instructions**: `bt`/`bts`/`btr`/`btc`, `popcnt`, `bsr`/`bsf`,
      `tzcnt`/`lzcnt`, `bswap`.
- [ ] **BMI/BMI2 when available**: `andn`, `bextr`, `blsi`/`blsr`/`blsmsk`,
      `shlx`/`shrx`/`sarx` (no flag clobber, no CL constraint), `mulx`.
      Needs a CPU feature level knob.
- [ ] **Carry-chain forms**: `adc`/`sbb`, and `setcc`-free branchless idioms
      (`sbb reg,reg` as a 0/-1 mask).
- [x] **Rotates.** `Or(Shl(x, k), Shr(x, w - k))` on a `w`-bit `x` becomes
      `rol k`: three instructions and two reads of `x` collapse to one of each.
      There is no `ror` form -- `ror k` is `rol (w - k)` in the same encoding
      size, and the operand order of the `Or` says nothing about which direction
      the source meant. `Shr` and not `Sar`, since an arithmetic shift feeds the
      sign bit into the high end. On `tests/lit/asm/rotate.c` a rotate function
      is 3 insts / 6 bytes against 9 / 19 for the same shape on a signed
      operand; no corpus row changed, because none of them rotates.
- [x] **Double shifts.** The rotate rule generalized: `Or(Shl(x, k), Shr(y, w - k))`
      on `w`-bit `x` and `y` is `rol k` where the two are the same value and
      `shld x, y, k` where they differ. There is no `shrd` form -- `shrd y, x, w - k`
      computes the same bits, so the two differ only in which operand the
      destructive form consumes and extraction has no basis to prefer either.
      On `tests/lit/asm/double_shift.c` a funnel shift is 3 insts / 7 bytes
      against 7 / 15 for the same shape on signed operands; no corpus row
      changed, because none of them funnel-shifts. `shld` is priced at Skylake's
      latency 3 / throughput 1, which loses under the `Latency` goal and wins
      under `Balanced` on its size and its single read of each operand.
- [x] **LHS-iconst compare commutation.** A constant left-hand operand moves
      right and the condition flips, so `X86CmpI` matches it and the constant
      is never materialized. Over the rows that changed, `lit` -0.54% insts and
      -0.40% bytes, `fuzz` -0.79%/-0.71%; `bench` and `live` unchanged.
      The swap is a normal form `IRBuilder::icmp` builds the node in, not an
      e-graph equality: an `Icmp`'s condition code is read from its e-class,
      independently of which node extraction picks, so two `Icmp` nodes with
      different codes in one class would make that read ambiguous.
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
      `docs/internal/refactor-roadmap.md` step 6: the fold is done and this is allocation
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

**No wrong-value programs are open. Sixteen capacity failures are**, found
by running the generator at 200 seeds a shape instead of the 30 every gate is
pinned at. The ones worth keeping are checked in under `tests/fuzz/corpus/open/`,
where `run_corpus.sh` re-checks them in seconds; the files are the durable
artifact and the seed is not, since `gen_c.py --seed N --shape S` only
regenerates a program until the generator changes.

| shape | passing | wrong value | capacity |
| --- | --- | --- | --- |
| `mixed` at 200 | 198/200 | -- | 57, 123 |
| `args` at 200 | 188/200 | -- | 12 seeds |
| `pressure` at 200 | 198/200 | -- | 98, 148 |

**Re-measure rather than trust the list.** Entries have left it without anyone
fixing them before, and one went the other way: `mixed` 57 is a capacity failure
step 6's fold introduced.

The last wrong-value bug closed was one defect behind all of them: slot routing
named a block parameter by the VReg the *class map* gave at block entry rather
than the one the block's own `Op::BlockParam` defines. Where those disagreed the
reloads went in front of uses of a VReg the block never mentions, so the block
kept reading a register no predecessor writes -- on `pressure` seed 14 an
inlined loop counter started at 14 instead of 0 and the loop was skipped
entirely. `BLITZ_DEBUG=paramsrc` prints exactly that disagreement,
`b41.p0 -> 252: schedule=252 map=167 cfg=252`, and is the first thing to run
when a parameter reads the wrong value.

The capacity failures are the two shapes the allocator names itself: *"spilling
did not reduce it, so the pressure point is one instruction whose own operands are
what is live there"*, and *"every over-budget VReg is a block parameter, which
only the splitter can route through a slot"*.

What the fixed ones are worth is the shape they kept having, which is the first
thing to check on any of these:

- **A block resolved an e-class to the wrong VReg** -- nine bugs, and the reason
  steps 1-4 of `docs/internal/refactor-roadmap.md` exist. `BLITZ_DEBUG=regalloc` dumps the
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

- [ ] `docs/internal/split-pass-plan.md` Phase 8 and Final Audit are unchecked. The audit
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
