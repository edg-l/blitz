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

## Current state (2026-07-25)

- 905 Rust tests + 390 lit tests, all green. `cargo fmt` clean.
- Two open bugs from the random generator, both in regalloc (see Known bugs).
- `BLITZ_VERIFY=1` and `BLITZ_VERIFY=strict` green across both suites.
- `bash tests/lit/run_diff.sh`: 262 tests compared O0-vs-O1 and against a
  reference compiler; no skips, no differences under gcc or clang.
- Pipeline: IR -> inlining -> DCE1 -> store/load forwarding -> DSE -> LICM ->
  e-graph saturation -> cost-based extraction -> DCE2 -> linearize -> DAG
  schedule -> live-range splitter -> function-scope Chaitin-Briggs regalloc ->
  terminator lowering -> MachInst lowering -> branch relaxation -> ELF.
- Implemented e-graph rules: see `docs/egraph-optimization-roadmap.md`.
- Splitter design: see `docs/split-pass-plan.md`.

## Priorities

### P0 -- Correctness

A fast wrong answer is worthless, and an optimizer this size certainly has
subtle bugs left. Every miscompile found so far was found by a hand-written
test that happened to hit the right shape (see the 04-18 regalloc fix wave,
`2c6cc8d`, `752ed7e`, `78850a2`). Hand-written tests are a weak oracle. The
priors on where bugs live: **regalloc** (splitting, coalescing, spill/remat,
cross-block liveness) first by a wide margin, then the **memory passes**
(forwarding/DSE resting on a conservative alias model), then **isel width and
type handling** (the `X86CmpI` `ty` bug was exactly this class).

- [x] **Differential execution over the lit corpus** (`tests/lit/run_diff.sh`).
      Compiles every runnable lit test at -O0 and -O1 and compares exit status
      and stdout; no expected output needed, only that optimization preserve
      behavior. Found a wrong-code bug on its first run.
- [x] **Random program generator** (`tests/fuzz/gen_c.py`, `run_fuzz.sh`).
      UB-free by construction, interprets as it generates so the expected
      output is known, and aims at what the corpus misses: 7-12 parameter
      functions past the argument registers, interleaved int/double
      signatures, and more live values than registers. Checks blitz against
      its own prediction, -O0 vs -O1, and `cc`.
- [ ] **Shrinking.** Failures currently come out at 40-3000 lines; reducing
      them to a minimal lit test is manual. Delta-debugging over the AST is
      the natural fit since the generator can re-simulate any candidate.
- [x] **A gcc/clang oracle in the harness.** O0-vs-O1 self-consistency cannot
      see a bug that is equally wrong at both levels -- exactly how the
      `cvtsi2sd` REX.W bug and the missing variadic `AL` survived.
      `run_diff.sh` now also compiles with `cc` and compares. Clean against
      both gcc and clang over 262 tests.
- [ ] **Reference IR interpreter.** The stronger oracle: execute the IR
      directly and compare against the compiled binary. Also lets a failure be
      attributed to a specific pass by re-running the interpreter on the IR
      after each stage.
- [x] **Per-pass IR verifier** behind `BLITZ_VERIFY=1` (`src/verify.rs`): block
      shape, unique block ids, edge targets, block-param arity and types,
      entry-block params, `Ret`/`Call` arity, and e-graph class resolvability.
      Green across all tests and lit.
- [x] **Canonicalize CFG class references after every merging pass**
      (`src/compile/canon.rs`). Effectful ops used to keep pre-merge `ClassId`s,
      leaving every consumer to canonicalize on read -- soundness-neutral until
      one forgets, which is what `ca2e400` was. `BLITZ_VERIFY=strict` is the
      standing acceptance test and is green.
- [~] **Machine-level verification.** Frame layout is covered: four properties
      (RSP 16-byte aligned at call sites, frame reserves spills + outgoing args,
      red-zone preconditions, spill area does not overlap outgoing args) are
      checked exhaustively over 12k configurations in `src/x86/abi.rs`. Still
      missing, and needing a `defs()`/`uses()` on `MachInst` first: no vreg
      survives the rewrite, no two overlapping live ranges share a physical
      register, callee-saved actually preserved.
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

- [ ] Benchmark harness: a corpus of C files, compiled by tinyc/blitz and by
      `gcc -O2` / `clang -O2`, comparing instruction count, `.text` bytes,
      spill/reload count, and wall-clock via hyperfine. Checked-in baselines so
      regressions are visible in a diff.
- [ ] Per-function codegen stats behind a flag (instructions, spills, reloads,
      stack frame size) so a pass's effect is one command away.

### P1 -- Optimizer gaps with the largest measured impact

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

Both found by `bash tests/fuzz/run_fuzz.sh`, which failed 8 of 8 programs on
its first run. Both are in the register allocator, matching the standing prior
that regalloc carries the highest bug density.

- [ ] **Splitter does not converge on real register pressure.** Overshoots of
      up to 18 GPR colors reach phase 5, which aborts compilation. Every
      remaining `run_fuzz.sh` failure is this one; no miscompiles are left in
      the generated corpus.

      Investigated: it does split only the worst point per block and never
      re-measures. Making it iterate (recompute pressure, split again) does
      help -- excess dropped 14 to 8 within a block -- but stalls, because
      `SplitScope::PerBlock` only considers values defined in the block, and
      these blocks are dominated by live-through values. Falling back to
      `SplitScope::CrossBlock` when a round retires nothing regressed 4 lit
      tests. Both experiments are reverted. The fix is not a local tweak: the
      splitter needs to be able to spill a live-through value from an arbitrary
      block, which means placing the store in the defining block and reloads at
      every use, driven by a real iterate-to-fixpoint loop.
- [ ] **Stack array access corrupts the frame**
      (`tests/fuzz/findings/array_spill_frame_corruption.c`, 12 lines; reduced
      from `seed5_miscompile.c`). Summing elements of an `int arr[8]`:
      5 elements is correct, 6 prints the right value but returns exit 2, and
      **Fixed** (`8acd79b`, `350d36a`, `386116e`) and now a live regression
      test: four defects compounded -- a stale addressing-mode fold, load/store
      addresses resolving to the pre-spill register, `return 0` dropped after a
      call, and a reload emitted ahead of the store to its slot.
- [x] **Rematerialized address emitted after the store that uses it** -- fixed
      in `f4d36ff`. The constraint-propagation fixpoint in
      `assign_barrier_groups` did not mark itself changed when it *created* a
      constraint, so propagation stopped one level short of the value that
      needed it. `seed5_miscompile.c` prints 606 at -O1, matching gcc, clang
      and the generator.
- [ ] **seed5 still segfaults at -O0.** The reloads write RAX while the loads
      read RCX: `mov rax,[rsp+0x30]` / `mov ecx,[rcx]`. A reload's destination
      register and the register its consumer reads have diverged. The -O1 path
      is fixed and the plain array case passes at -O0.
      Narrowed by pass bisection: `-O0 --enable-inlining` **passes**, every
      other single pass still fails. So it needs `f0` to remain a real call --
      mixed int/double arguments with values live across it -- rather than
      being inlined away.

## Tooling to build next

Ranked by what actually cost time while fixing the six bugs above, not by
generality.

- [x] **Machine-level verifier over the final MachInst stream** (`8d99493`).
      `MachInst::defs()`/`uses()` for all 79 variants, plus forward dataflow
      over the CFG recovered from labels and branches, meeting predecessors
      with intersection. Reports a read of a register not written on some path
      to it, and any surviving virtual register. Runs under `BLITZ_VERIFY`
      after branch relaxation; green everywhere.
      Known limit: it cannot see a register that holds the *wrong* value, which
      is what seed5 does at -O0. Value errors stay the differential harness's
      job.
- [ ] **Extend it to spill slots and callee-saved registers**: a reload must
      not read a slot never stored (same dataflow, keyed by frame offset), and
      a callee-saved register written in the body must be saved and restored.
- [ ] **Splitter/allocator agreement assertion.** Assert the splitter's
      post-plan pressure matches the allocator's chromatic number and dump the
      disagreeing point. These two already disagreed silently once
      (per-block vs function-wide `vreg_classes`) and that was a real bug.
- [ ] **`BLITZ_DEBUG=split`.** Promised in `docs/split-pass-plan.md`, never
      implemented; the split plan (victims, slots, insertion points, segments)
      had to be re-derived with throwaway `eprintln!`s four separate times.
- [ ] **Delta-debugging in the fuzzer.** Failures arrive at 40-3000 lines and
      were reduced by hand twice. The generator can re-simulate any candidate
      to confirm it is still UB-free and still failing, which is exactly what
      automated shrinking needs.

## Tech debt

- [ ] `docs/split-pass-plan.md` Phase 8 and Final Audit are unchecked. The audit
      requires zero hits for `coalesce_aliases`; there are 17 across
      `regalloc/mod.rs`, `global_allocator.rs`, `compile/terminator.rs`,
      `compile/mod.rs`. Decide whether it is now load-bearing, then finish or
      amend the plan.
- [ ] `src/compile/mod.rs` is ~1600 lines; split it.
- [ ] Clear the remaining clippy backlog (~29 warnings, cosmetic).
- [ ] `README.md` says 315 tests and omits forwarding/DSE/splitter from the
      pipeline diagram.
- [ ] Handoffs live in gitignored `.claude/handoff/` and are not in Engram, so
      session continuity is invisible to a fresh agent.

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
