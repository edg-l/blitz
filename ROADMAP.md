# Blitz Roadmap

## Goal

Produce the best possible x86-64 machine code. Blitz targets **one** ISA on
purpose: the optimizer can reason natively about addressing modes, flags, LEA,
multi-output instructions, and uarch behavior instead of laundering everything
through a target-neutral IR. Anything a portable backend has to generalize away
is fair game here.

The measure of success is how fast the emitted code runs against `gcc -O2` /
`clang -O2` on the same input, in CPU cycles.

**Where that stands: `x3.07` against `gcc -O2` and `x3.80` against `clang -O2`**,
geometric mean of the per-program cycle ratio over the 21 `live` kernels, from
`bash tests/run_perf.sh`. Worst is `byte_copy` at `x17.6`.

Read `live`, not `bench`. Both exist, but `bench`'s kernels resist folding only
by being large enough that `gcc` gives up. Every `live` kernel seeds its data
from `argc`, which no reference compiler can evaluate, and takes its repeat
count from `argc` too, so the harness scales the work by passing arguments.

**Instruction counts are diagnostics, not the ranking, and this was measured
rather than assumed.** `run_codesize.sh --gap` reads `x1.29` on the same corpus,
which is 2.4x better than the truth. Two independent reasons, and both matter:

- **A static count is unweighted by execution frequency.** A loop body run a
  million times counts once, so a corpus of loop kernels -- which is what `live`
  is -- reports mostly the quality of its own straight-line setup code.
- **Instruction counts invert on latency-for-instruction trades**, which are
  trades worth making. `x * 7` as `shl; sub` retires one *more* instruction than
  `imul` and costs `1.1%` fewer cycles; gcc makes that trade routinely. A metric
  that scores it as a regression cannot rank a compiler against gcc.

The second is why the `clang` column used to need a disclaimer: static counting
had blitz *ahead* of `clang` at `x0.78`, because `clang` unrolls and vectorizes
into more instructions that run faster. It is 3.8x faster, and the number now
says so instead of needing a paragraph explaining when it means the opposite.

**IPC is not a goal either.** `run_perf.sh` prints it because it says *why* a
kernel is slow -- high IPC with high cycles is too much work, low IPC with high
cycles is stalls -- not because more is better. A compiler emitting more, cheaper
instructions raises IPC while doing more work.

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

**`P0`..`P4` under Priorities is the only numbering in this document.** This
section is the queue: what to pick up next, in order, each entry naming its tier
rather than restating it. Take them top to bottom.

- **Loop headers are aligned by accident** -- `P1`. Function starts are aligned
  now, so a change in one function no longer re-times another; the loop headers
  themselves still are not, and `align_loop_headers` needs align and relax
  iterated to a fixpoint before it can be wired. Worth up to 20% on one kernel:
  removing 4 dead instructions from `matmul_rt` cost `+20.7%` cycles, and three
  unrelated instructions before its loop took the same comparison to `-1.0%`.
- **Copies are still a third of what blitz emits** -- `P1`.
  697 register-to-register moves in 2407 instructions over `bench` against
  `gcc -O2`'s 351 in 2322, and **585 of the 697 are parallel copies** -- phi
  copies on edges, entry parameter moves, argument setup -- which on their own
  exceed gcc's entire copy count. `stats` reports the split and
  `run_codesize.sh` tracks it. Read `docs/internal/refactor-roadmap.md` before
  starting.

`P1` is ordered by measured impact. **`P2` is where the
single-target thesis pays off**, though half of it is unreachable until tinyc
grows builtins -- see the note there.

**One reordering worth stating outright.** `P2`'s constant cost model belongs
above most of `P1`: it unblocks three isel items at once, the missing structure
is a parents map in the e-graph, and the payoff is measured. The temptation is
to start on GVN or loop strength reduction because they are the recognizable
names, and neither is where this backend is currently losing -- `P1`'s own
copies item says the whole remaining instruction gap to gcc is copies.

**Do not start in the register allocator, the splitter, or the block-parameter
machinery without reading `docs/internal/refactor-roadmap.md` first.** It is
finished as work and is now the record of what eleven steps measured --
including the predictions that were wrong, which is what stops the next attempt
repeating them.

## Closed, and what they measured

These stay because what they measured is the reason the next attempt should not
start where they did.

### ~~Finish the `-O0` allocator~~

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

### ~~The last capacity failure~~

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

**This is the same shape as the `-O0` allocator's third step**, arrived at from
the other end: a pre-pass relieving pressure against a different graph than the
allocator enforces. It was under-relieving here rather than over-relieving.

### ~~The allocator's liveness disagrees with the emitted code's~~

**It did not.** `verify_register_sharing` was reading one of its inputs from a
copy taken before the allocator's spill loop, and `BLITZ_VERIFY` is green at
both `1` and `strict` across all 560 lit tests now that it derives that input
itself.

The input is the terminator half of liveness -- what each block's terminator
passes to its successors. `barrier::terminator_uses` states the rule in its own
doc comment: the set is a function of the schedules at the moment it is asked,
because every pass that rewrites an operand rewrites it there. `main` in the
reproducer needs one spill round; that round rematerialized a double a terminator
passed and renamed the argument, so the pre-loop copy named a VReg whose real
range ends at its `XmmSpillStore`. Carried to the block exit instead, it read as
live across the whole tail of block 2, and the value that legitimately held
XMM13 there read as a clash. `allocate_global` had already been bitten by exactly
this and recomputes the set every round; the verifier took the stale one.

**What it cost to find**: one probe, printing the reported pair's defs, uses and
membership in both the stale and the freshly-derived set. Two lines of output
settled it -- `def block 2 [148]`, `use block 2 [149] Pseudo(XmmSpillStore(193))`,
`stale phi_uses[2] names it`, and no final `phi_uses` naming it at all.

**The fix is the parameter's removal**, not a check on it. A caller cannot pass a
stale set to a function that does not take one.

**The lesson is the one the entry above it already carried**, arrived at from the
verifier's side: a *check* built on a snapshot is as wrong as a *pass* built on
one, and it is worse, because a red gate with no behavioural symptom is read as a
bug in the component it names. This entry sat at the top of `P0` for three
sessions as an allocator bug. Related sub-item it also carried, now stale: the
pre-coloring conflict assertion exists (`coloring::check_precolorings`, under
`BLITZ_VERIFY`), and `interval_color` no longer exists to need one.

## Current state (2026-08-07)

- 1010 Rust tests + 552 lit tests, all green. `cargo fmt` clean, `cargo clippy
  --all-targets` clean, zero build warnings, zero rustdoc warnings.
- `BLITZ_VERIFY=1` and `BLITZ_VERIFY=strict` green across both suites, with no
  row red on purpose. **No `P0` item is in the Start here queue any more**: the
  queue starts at `P1`. `P0` still holds the items below that no reproducer names
  -- the `assign_args` panic, the C-surface probe, the second implementation, the
  one-fact-one-place audit.
- `bash tests/lit/run_diff.sh`: 337 compared `-O0`-vs-`-O1` and against a
  reference compiler; no skips, no differences under gcc or clang.
- **`-O0` is on `regalloc::fast` and both levels are correct.** Everything below
  is `-O1` unless it says otherwise.
- `bash tests/fuzz/run_corpus.sh`: 17 `fixed` programs, all passing at both
  levels, and nothing open.
- Generated programs: `mixed` 400/400, `args` 400/400, `pressure` 400/400, which
  is what `run_fuzz.sh` now sweeps by default, plus `abi` -- 98 programs
  enumerating the argument surface rather than sampling it, all passing. **The width is what makes it a
  check** -- the `-O1` allocator bug in `fixed/args-seed310.c` is at seed 310 of
  `args` alone, and at the 30 seeds the gates used to be run at, all three
  shapes were green while seven programs miscompiled.
- Code quality has a baseline: `bash tests/run_codesize.sh --check`, over `lit`,
  `bench`, `live` and `fuzz`, fed by `BLITZ_DEBUG=stats`.
- Code quality also has an *absolute* number, which is the one the Goal is
  written against: `bash tests/run_perf.sh`, `x3.07` cycles vs `gcc -O2` over
  the 21 `live` kernels, median of 5 `perf stat` samples each. Cycle counts vary
  ~1% run to run here; instructions retired vary 0.00%, and are the wrong metric
  for the reason the Goal gives. Widening `live` has no natural ceiling and is
  always a valid use of leftover time.
- `--gap` still prints the static instruction and byte ratios and is still
  worth reading -- it is how a folded program is *detected*, since `lit` and
  `fuzz` compute a fixed answer from no runtime input and `gcc -O2` emits the
  constant. It is a diagnostic. It is not the ranking, and it read `x1.29` where
  the truth was `x3.07`.
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
- **What the e-graph is actually doing**, since the pipeline line above makes it
  look like the optimizer: it is the *instruction selector*, which is where
  competing forms genuinely exist and a cost model has to choose between them,
  plus a cleanup rewriter. The equality-saturation part -- iterating to a
  fixpoint while holding every alternative -- is worth 0.39%. Do not go looking
  for wins in there; see Decisions, and the exploration-rule experiment in P1
  for the one way that could change.
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

**Open correctness work, before any diagnostic.**

- [x] **An odd number of stack arguments misaligned the stack.** SysV wants
      `RSP % 16 == 0` at a `call` and `setup_call_args` emits one `push` per
      stack argument, so seven integer arguments left the callee eight out and
      its own call into libc faulted. Even counts survived by luck, which is why
      nothing caught it: the generator's `args` shape and every corpus program
      land on even counts. `abi::stack_arg_bytes` now owns the number the setup
      and the cleanup both move RSP by, padding an odd count. Costs one `sub` per
      such call site -- 50 rows of `lit` and `fuzz` at +0.0% to +0.4%, with
      `bench` and `live` unmoved. `corpus/fixed/stack_arg_alignment.c` and
      `lit/functions/stack_arg_alignment_odd.c`.
- [x] **Entry parameter moves were a sequence, not a parallel copy.** A
      parameter in a caller-saved register gets no pre-coloring when its block
      contains a call (`precolor::assign_param_vregs_from_map`), so it arrives by
      an entry move instead -- and the two lists of those moves were emitted in
      order, letting `mov rcx, rdi` destroy the fourth argument before the move
      that reads it. Six arguments summed to `a+b+c+a+c+f`. The same shape at
      `-O0` gave every parameter the first free register, since `pick` starts
      from an empty `taken` at each instruction, and stored one value into all
      six slots. Both are now the one fact they always were: a parameter is in
      its argument register before the function's first instruction runs.
      `lit/regalloc/entry_param_parallel_copy.c`. **Found by reading the values
      out of a program the segfault above was hiding** -- 546 lit tests, 335
      differential comparisons and 400 seeds a shape were all green while a
      six-argument function that calls anything computed the wrong sum.
- [x] **Every parameter of a function was modelled as holding a register at
      entry.** True of a block parameter, whose phi copies wrote it on the edge,
      and of a register-passed function parameter, whose caller did. False of a
      stack-passed one: its value is in the caller's frame and its marker *is* the
      load. `add_param_interferences` and the splitter's two pressure scans both
      asserted it of all of them, so 15 integer parameters were a clique of 15
      where 14 colours exist and `callee` did not compile at `-O1` -- at a
      measured peak of 8 GPRs live, so the graph and the pressure disagreed by
      seven. Fourteen fitted exactly and hid it. `abi::marker_is_entry_resident`
      is now the one place that decides, and the splitter and the colourer read
      the same one -- they have to, or the colourer needs a register the splitter
      was never asked to free. `lit/functions/fifteen_int_params.c`. Found by the
      ABI enumeration; changed none of the 980 codesize rows.
- [x] **A non-`static` function with no caller in its own file was not emitted**,
      so separate compilation produced an object nobody could link against. Two
      files, `helper` defined in one and called from the other:

      ```
      /usr/bin/ld: b.o: in function `forward':
      (.text+0x18): undefined reference to `helper'
      ```

      `readelf -sW` on the object showed `main` alone where `cc` emits `main` and
      `helper`. Nothing in that translation unit calls `helper`, so the
      dead-function elimination that `lit/inline/dead_func_eliminated.c` covers
      removed it -- right for a whole program, wrong for a `-c` compile, where any
      external definition may be the one another object needs.

      **The condition it was gated on was `has_main`, and that is a fact about the
      module rather than about the compilation.** `CompileOptions::whole_program`
      is the fact that was missing, it defaults to `false`, and only the driver
      sets it: a module is the whole program when it is the sole input *and* this
      run produces the executable. That covers `-c` and the multi-input link too,
      which fails identically -- `tinyc a.c b.c` compiles each file separately and
      links the objects afterwards, so neither is a whole program either.
      `has_main` is now checked inside `eliminate_dead_functions`, next to the BFS
      that needs a root, rather than derived by two callers.

      **What it costs**: a `static` helper is kept as well, because `ir::Function`
      carries no visibility and nothing can tell the two apart. Recovering that
      needs `static` in tinyc and a linkage field on `Function`. Note also that
      elimination is still reached only through `inline_module`, so it does not run
      at `-O0` at all.

      `lit/multifile/cross_file_uncalled_definition.c` and the tinyc unit test
      `test_uncalled_definition_survives_separate_compilation` pin both halves.
      Found while checking that a cross-object *tail* call links. It has nothing to
      do with tail calls -- the non-tail version failed identically -- and it is the
      kind of defect no single-file corpus can see, which is the same gap the
      multi-file tests exist for and did not cover: every one of them puts `main`
      in the file under test and the helpers in a file that has none, so
      `has_main` was false there and the elimination never ran.
- [ ] **`assign_args` `unreachable!()`s on a struct** (`src/x86/abi.rs`). Same
      tier mistake: the frontend cannot produce one today, so it reads as a
      feature gap, but the failure mode is a panic rather than a diagnostic.
- [ ] **Probe the C surface the way the ABI surface should be probed.** Twenty
      small programs run against `cc` found five failures, and the four that
      were silent wrong answers were worth more than the whole session's
      guessing: block scoping, pointer difference, an unsigned compare whose
      condition was flipped out from under it, and a logical shift folded on a
      sign-extended pattern. The loud ones -- a missing `sizeof(expr)` -- cost
      nothing, because a parse error cannot be mistaken for a result. **tinyc is
      inside every oracle's trusted base**: `run_diff.sh` compares blitz against
      `cc` on source tinyc parsed, so a frontend bug is reported as a blitz
      difference and costs a session in the wrong component. This is the same
      item as the ABI enumeration above, moved up a level from calling
      conventions to the language.
- [ ] **A second implementation of the pass with the worst bug rate.**
      `regalloc::fast` was built for DWARF and turned out to be the highest-yield
      bug finder in the project: it found an `-O1` allocation bug within hours of
      being correct, and `BLITZ_VERIFY` over it found two more. That is ~350
      lines. The same is available for the scheduler (source order, no DAG) and
      the extractor (greedy, no cost model), and each would make a whole pass's
      bugs a disagreement `run_diff.sh` can see rather than an answer both levels
      give. **Treat this as a standing strategy, not the one-off it looks like.**
- [x] **Enumerate the ABI surface rather than sampling it.** `tests/fuzz/gen_abi.py`
      walks counts 0..25 x {int, double, mixed} x {leaf, calls printf}: 152
      programs, 7.8s, the fourth shape of `run_fuzz.sh` rather than a fifth gate.
      **The ceiling is past the register file on purpose**: the first run stopped
      at 16 and both defects it found were at 14 and 15, at the top of the range,
      which is the signature of a bound hiding something rather than one that has
      been reached. 17 through 25 are clean at both levels and under
      `BLITZ_VERIFY=strict` -- 19 integer arguments on the stack -- so the two
      fixes generalise rather than being off-by-one patches at the sizes that
      exposed them.
      A failing program *names the argument* -- the callee returns the 1-based
      index of the first one that did not arrive. **Three defects on the first
      run**, none of them reachable from `gen_c.py`, which caps at 12 parameters:
      a silently wrong first double at `-O0` from nine parameters up (the entry
      sequence scratched XMM0 while it still held one), a call of 14 arguments
      that could not be allocated at either level, and 15 parameters that could
      not be coloured because every parameter of a function was modelled as
      holding a register at entry. **All three fixed, and 98/98 passes under
      `BLITZ_VERIFY=strict`.** None of the three changed a single one of the 980
      `run_codesize.sh` rows, which is the measure of how far outside the sampled
      space they were.
- [ ] **A "one fact, one place" audit.** Two bugs in one session were the same
      fact derived twice and disagreeing -- the block's `param_vregs` against
      `cfg::resolve_block_param_vreg`, and `Op::BlockParam`'s shadow modelled
      while `Op::Param`'s was not. The repo already knows this pattern:
      `EffOperand` and `TermArgs` were exactly this refactor, and "a block
      resolved an e-class to the wrong VReg" is nine bugs. There is no item for
      finding the instances that are left.

**Diagnostics worth building, each earned by a session it would have shortened.**
None is a gate; they are what turns a wall of numbers into a name. **Kept
deliberately short.** Twelve unbuilt diagnostics is a backlog that never drains,
and neither of the two things that actually found bugs recently -- a second
implementation, and a wider sweep -- was on that list. Build the rest the next
time an absence costs a session.

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
- [ ] **Reference IR interpreter.** The stronger oracle: execute the IR
      directly and compare against the compiled binary. Also lets a failure be
      attributed to a specific pass by re-running the interpreter on the IR
      after each stage.
- [ ] **The rest, unranked and not scheduled**: naming the pass behind each
      `--check` regression, a decision diff rather than an output diff,
      callee-saved preservation at machine level, a stronger UB guard in
      `reduce.py`, rewrite-rule equivalence tests, a regalloc stress mode, a
      csmith-lite for tinyc, and every fuzz find landing as a lit test. Each is
      justified; none has been paid for by a session recently enough to rank.

### P0 -- Measurement

Without numbers, "most optimized" is unfalsifiable and every session drifts.

**What exists**: `bash tests/run_codesize.sh [--check|--update|--gap]` over four
corpora with a baseline each in `tests/baselines/`, fed by `BLITZ_DEBUG=stats`.
Instruction count, `.text` bytes, spill stores and reloads per (program, level);
a generated program that does not compile is a `-` row rather than an omission,
so the holes stay visible.

- [x] **Measure how fast the code actually runs.** `bash tests/run_perf.sh`:
      `perf stat` cycles for blitz and each reference on the `live` kernels,
      median of 5, ranked by cycle ratio. This was deferred on the grounds that
      blitz-against-itself had refereed every decision so far -- and that is
      exactly what went wrong, since blitz-against-itself cannot see that its own
      yardstick is 2.4x optimistic.
      **Wall clock was the wrong instrument, not the wrong question**: it reads
      0.9-45% run to run here against cycles' ~1%, because it measures the
      scheduler as much as the code. Counters were free and already installed.
      The kernels each take their repeat count from `argc`, so an ordinary run
      does a hundredth of the work and the lit and differential harnesses stay
      quick.

### P1 -- Optimizer gaps with the largest measured impact

- [ ] **Copies are still a third of what blitz emits.** Re-measured 2026-08-07
      over the 15 `bench` kernels: blitz **697 register-to-register moves in 2407
      instructions** (29.0%), against `gcc -O2`'s 351 in 2322 (15.1%) and
      `clang -O2`'s 259 in 2661 (9.7%).

      **The copy surplus is now four times the whole instruction gap**: blitz is
      85 instructions behind gcc and emits 346 more copies than it, so removing
      them would put blitz ahead on the count outright.

      **And the split says which fix applies.** `BLITZ_DEBUG=stats` now reports
      `copies=` and `two_addr=` per function, and `run_codesize.sh` tracks
      `copies` as a fifth baseline column, so any change here is measured rather
      than argued. Over `bench`: **112 two-address fixups and 585 parallel
      copies**. The two-address half is already down from the 246 that
      `coloring.rs` documents, which is `two_address_hints` working; the
      remaining 585 are phi copies on edges, entry parameter moves and argument
      setup, and **blitz's parallel copies alone exceed gcc's entire copy
      count.** That is where the item is.
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
- [ ] **Dominance-scoped elaboration, which is GVN and several other things at
      once.** The e-graph does local CSE only, so repeated address computations
      and field loads survive across blocks -- typically 5-15% on real code. But
      blitz does worse than miss them: linearization *re-emits* a class in every
      block whose uses the original definition does not reach, which is the
      anti-GVN.

      Cranelift gets GVN for free instead, by rebuilding SSA in dominator-tree
      order with layered scope maps and computing each value on demand in the
      scope that needs it (Fallin, 2026 -- the aegraph retrospective). One
      change would collapse this item, delete the per-block re-emission
      machinery, and shrink the "one class maps to several VRegs" hazard that
      has produced nine wrong-code bugs here.

      **Invasive, and it lands squarely in what
      `docs/internal/refactor-roadmap.md` warns about.** Read that first. It is
      still the single largest take from an outside design that blitz has not
      already arrived at independently.
- [ ] **Loop strength reduction + induction variable recognition.** Every array
      loop recomputes `base + i*scale`. Worth 2-5x on loop-heavy code and is
      table stakes for calling the backend "optimizing".
- [ ] **SCCP.** `propagate_block_params` only handles single-predecessor
      constants; conditional arms that become constant are missed.

      **The constant meet is written and measured, and it cannot land as
      written.** Extending the pass from one predecessor to a meet over all of
      them -- a parameter every predecessor passes the same constant to *is* that
      constant -- is `lit` `-5.2%` instructions over 33 changed rows and `fuzz`
      `-1.3%` over 88, with `bench` and `live` unchanged. It also cuts copies,
      which is the item above: `inline/inline_multi_return.c` goes from 9 to 1.

      **It introduces an `-O0` miscompile, and the cause is the merge rather than
      the meet.** `run_diff.sh` catches it on `control/block_scope_shadowing.c`:
      `-O0` prints 0 where `cc` and `-O1` print 7. Merging the parameter's class
      with the constant's leaves a class holding *both* `BlockParam(b3, 0)` and
      `Iconst(7)`, and `cfg::resolve_block_param_vreg` asks the target block's own
      `BlockParam` first -- so the use reads the parameter's slot, whose phi copy
      sources that same merged class and therefore copies the slot from itself.
      The slot holds 0.

      Ruled out on the way, so the next attempt does not re-check them: a
      constant class used in a block its definition does not dominate is
      *correctly* re-emitted at `-O0`, both as a `Ret` value and as a call
      argument (`if/else` where both arms print the same literal is fine). The
      hazard is specific to a class that is also a block parameter.

      **So the route is `phi_removal`'s protocol, not a merge.** A parameter the
      meet proves constant has to be *removed* from the block and its edges, then
      re-extracted and re-linearized, and the result verified -- which is exactly
      what `phi_removal` already does, and why it "proves its own result rather
      than predicting it" after two attempts to predict merge composition were
      both defeated by a program. **The existing single-predecessor merge has the
      same latent hazard**; nothing in the current corpus reaches it.
- [ ] **Memory SSA / memory versioning.** Makes forwarding, DSE, and GVN work
      cross-block on shared machinery instead of three intra-block passes.
- [ ] **nsw/nuw/nnan/ninf op flags.** Without them the signed-ordering
      algebraic rewrites stay permanently rejected (see Decisions). Op-flag
      bitfield threaded through saturation.
- [x] **Tail call optimization.**
      A tail self-call is now a `jmp` to the label bound *after* the prologue:
      arguments into their ABI registers exactly as a call needs them, and control
      to the top of the body. The frame is neither torn down nor rebuilt, RSP does
      not move, and the body begins by moving each parameter out of its argument
      register -- which is the state a fresh entry would be in. No `call` means no
      return address pushed, so the base case's `ret` returns to the original
      caller.

      `tests/lit/live/tail_recursion.c` exists to price it, because nothing could:
      `bench`, `live` and the generated programs contained **zero** tail-call sites
      between them, and the 59 in `lit` are almost all `main` returning a call
      once. Median of 5 at `ARGS=100`:

      ```
      blitz -O1, no TCO        146.6M cycles    x2.87 vs gcc
      blitz -O1, self only     107.8M           x2.11            -26.5%
      blitz -O1, self + other   72.5M           x1.42            -50.6%
      gcc -O2                   51.1M
      clang -O2                 16.4M
      ```

      **Both forms, and they differ in one thing that decides everything else:
      whether the frame this function built is the one the callee will run in.** A
      self-call jumps to the label bound after the prologue and the frame stands.
      A call to another function tears the frame down first -- the callee's
      prologue builds its own -- so it is `setup_call_args`, then
      `abi::emit_frame_teardown`, then `jmp <symbol>` as a new
      `MachInst::TailCallDirect` (`E9 + rel32`, the same PLT32 relocation a call
      uses). RSP is then back on the return address the original `call` pushed, so
      the callee returns straight past this function. The argument registers
      survive the teardown because they are caller-saved and the teardown only pops
      callee-saved ones.

      `emit_frame_teardown` is the epilogue *without* the `ret`, split out of
      `emit_epilogue` rather than duplicated: the two have to move RSP by the same
      amount or the callee reads its return address from the wrong place, and
      nothing downstream could see that. Getting it wrong the first way was
      visible immediately -- the epilogue emitted its `ret` before the jump, so the
      function returned instead of tail-calling and the kernel printed 140106
      instead of 264767.

      **A static count cannot see most of this**, which makes it the sharpest
      example in the repo of why the ranking is cycles: a `call` becomes one
      `jmp`, so the self-call form moved `-0.8%` instructions for `-26.5%` cycles.
      The win is that a recursion deeper than the return-address predictor makes
      every `ret` mispredict, and this removes the pair.

      29 `lit` rows do improve on instructions, at `-3.7%` to `-16.7%` -- every
      `return f(x)` in the corpus loses its call and its `ret`. `bench` and the
      generated programs do not move at all; they discard no call results and
      return no call directly.

      `BLITZ_PASSES=-tail-calls` turns it off; it is on at `-O1` and off at `-O0`,
      where the recursion's frames are what a debugger walks.
      `lit/functions/tail_self_call.c` pins it, with `n * f(n - 1)` as the control
      that must keep its call.

- [ ] **Loop headers are aligned by accident, and it is worth up to 20%.**
      `emit::align_loop_headers` is written, exported from `emit/mod.rs`, and
      **never called**; `enable_nop_alignment` is `false` at both levels and is
      never read. So where a hot loop falls relative to a fetch boundary is
      whatever the instruction stream happens to produce.

      **Function starts are aligned now, which is the half of this that had no
      hazard.** `compile_module_with_globals` pads to 16 between functions, and
      what that buys is not speed -- `+0.04%` geometric mean over 24 `live`
      kernels, which is nothing -- but *invariance*: a function's offsets modulo 16
      no longer depend on the length of anything before it, so a codegen change in
      one function cannot re-time the loops of another. Demonstrated rather than
      argued: growing the first of four functions used to shift every absolute
      address in the last one modulo 16 (`14 15 0 2 4 ...` became `9 10 11 13
      15 ...`) and now leaves them identical. `compile::module`'s
      `function_starts_are_16_byte_aligned` pins it.

      **What is left is the loop headers, and the ordering is the problem.** NOPs
      have to go in before branch relaxation, or a jump the relaxer shortened may
      no longer reach; but relaxation then shrinks jumps and moves the header it
      just aligned. It needs align and relax iterated to a fixpoint, which is what
      an assembler does and what this pass does not. Two other things it gets
      wrong as written: the offset it computes is relative to the *body*, while the
      prologue is encoded separately and is part of the distance, and it never asks
      whether a loop is hot enough to be worth the padding.

      **Measured, and the measurement is the point.** Removing 4 dead `lea`s from
      `live/matmul_rt.c` -- a strict reduction, `-107M` dynamic instructions --
      cost `+94M` cycles, `+20.7%`. Not the frontend (idle went 0.29% to 0.67%, a
      rounding error against 20%), not branches (218.3M and 0.10% missed in both),
      and IPC fell 5.99 to 4.77. Then adding three unrelated instructions before
      the loop, changing nothing but where the code sits, took the same comparison
      from `x1.2069` to `x0.9901`.

      **This puts a layout term in every number the perf harness prints**, and
      nobody controls it. Two consequences: a pass that changes code size can be
      credited or blamed for up to 20% that has nothing to do with it, and
      `run_perf.sh` results are not comparable across changes that move code. Wire
      the pass up, then re-measure anything the layout could have decided.
- [ ] **Loop unrolling.** Compounds with LSR; do it after.
- [x] **Dead instructions in the final stream.** `emit::dead_inst` deletes
      instructions that compute a value nothing reads, over backward register
      liveness on the CFG `verify` already recovers from labels and branches.

      **Why it has to be here.** Lowering folds an address into the addressing
      mode of the load that uses it and decides that per consumer, at the last
      possible moment; when every consumer folds, the `lea` is left with nothing
      reading it. No earlier pass can see that -- DCE runs on the CFG before
      scheduling and the e-graph never sees an effectful op. Measured before the
      pass: 122 of 7801 instructions over `bench` and `live` were a `lea`
      immediately followed by an instruction naming the same address in its own
      mode, in every one of the 34 kernels, and they sit in loop bodies.

      ```
      bench   insts -2.69%   copies -1.80%    over 14 changed rows
      live    insts -1.33%   copies -2.67%    over 19
      fuzz    insts -0.40%   copies -1.13%    over 54
      lit     insts -0.98%   copies -1.35%    over 44
      ```

      **Zero regressed rows in any of the four corpora.** Cycles over `live`:
      `-0.85%` geometric mean, or `-1.69%` excluding `matmul_rt`, whose `+20.7%`
      is the layout artifact the item above is about. Six kernels gain more than
      1%: `histogram` `-11.8%`, `nested_carried` `-9.1%`, `pointer_chase` `-5.4%`,
      `branchy_filter` `-4.4%`, `byte_copy` `-3.7%`, `state_machine` `-1.9%`.

      What it will delete is deliberately short: register-to-register moves,
      immediate loads and `lea`, and nothing else. Not arithmetic, because EFLAGS
      is not a register this liveness models; not anything touching memory; not a
      write to RSP or RBP. `BLITZ_PASSES=-dead-insts` turns it off.

      **The bug worth remembering**: `MachInst::CallDirect` carries a symbol and no
      operands, so `uses()` says a call reads nothing -- and the `mov`s putting
      arguments in their ABI registers were dead. That deleted the argument setup
      of every call in the corpus, 248 of 576 lit tests, and is why `call_reads`
      exists.
- [ ] **Narrowing / type-width analysis.** **Attempted, measured, reverted, and
      the reason is a cost-model hazard worth knowing before the next attempt.**

      The concrete instance: **178 of 7696 instructions over `bench` and `live` are
      a sign-extension**, in every one of the 34 kernels, inside loop bodies -- an
      array index is `i32` in C and an address is 64 bits, so every subscript pays
      one. A `movsxd` of a value whose sign bit is known zero is a *zero*-extension,
      and `X86Movzx{I32, I64}` already lowers to **nothing** when its registers
      coincide (`lower.rs`) and to a 2-byte `mov` when they do not.

      The rewrite is 20 lines in `known_bits.rs` -- `Sext(x) -> Zext(x)` when
      `known_zeros` covers the sign bit -- and it is sound. It fired **nowhere**:
      178 of 7696 before and after. Every index in these kernels is a loop-carried
      value, and known-bits gives up on block parameters, so no loop index is ever
      provably non-negative. On a masked index (`argc & 7`) the rule does fire, 3
      `movsxd` to 1.

      **Then pricing the free widening as free regressed 234 rows across all four
      corpora** -- 165 of 180 `fuzz`, 58 `lit`, 5 `bench`, 6 `live`, everything
      that changed got worse. `X86Movsx` and `X86Movzx` are priced identically, so
      extraction had no reason to prefer the one that can vanish; setting the
      32-to-64 zero-extension to latency 0, size 1 made it prefer it, and made
      every class containing one look cheap. **A near-zero-cost node is an
      attractor**: extraction pulls subtrees through it that it should not.

      So the order of work is fixed, and it is not this: known-bits has to reach
      loop-carried values first, or the rule has nothing to fire on; and the cost
      model has to be able to say "free *only* when the registers coincide", which
      is a statement about allocation that a per-node cost cannot make. That second
      one is `P2`'s cost-model item again, the same prerequisite the
      exploration-rule experiment ended on. `(uint8_t)x + 1` should not promote
      to i32. Domain: `(min_bits, signed)` per e-class.
- [x] **Dead call elimination** for provably pure functions.
      `dce::pure_functions` is the greatest fixpoint over the module -- a function
      is pure when it stores nothing and calls nothing impure, so a self-recursive
      one stays pure and every extern is impure because nothing here can see into
      it. `eliminate_dead_pure_calls` then removes a call whose results no e-node
      and no other effectful op reads. Both halves of that are needed: the e-graph
      holds the pure computations and the CFG holds the effectful ones.

      **It removes the one dead computation nothing else could.** An effectful op
      is invisible to the e-graph by construction, so a call the inliner declined
      and whose result went unread survived every pass. Measured: 3 `fuzz` rows
      at `-3.1%`, `-5.2%` and `-4.0%` instructions with copies down with them,
      `bench` and `live` untouched -- those corpora discard no call results.
      Gated with the dead-load half, and for the same reason: removing a call
      takes away something a debugger could step into, which is the line `-O0`
      holds. `lit/functions/dead_pure_call.c`.

      It found a stale test on the way: `zero_arg_void_mixed.c`'s `void nop()
      { return; }` is pure, its result list is empty, and all nine calls to it
      were removed -- correctly, and leaving the test asserting nothing about the
      call-point detection it was written for. Its callee now stores.
### P2 -- x86-64 specialization (the differentiator)

This is where single-target focus is supposed to pay off. LLVM has ~10x the
isel patterns; we should beat it on the ones we implement.

**A non-goal blocks half of this tier.** The bit instructions, BMI and the carry
chain are not blocked on encoding but on *tinyc having no builtins*, and tinyc
is declared a test consumer rather than a product -- so under the current plan
they can never become reachable. They are marked below rather than ranked. The
cheapest unblock for the whole group is a small set of builtins in tinyc, which
is a decision to take deliberately or not at all; picking one of these items up
without taking it is picking up work that cannot land.

- [ ] **A cost model that can see a constant being materialized.** This blocks
      three separate items, which is why it leads the list -- and with half this
      tier unreachable it is the highest ratio of unblocks-to-cost in the whole
      document, not just here. `Iconst` costs 0.0,
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
- [ ] **A `subsume` marker on rewrite rules**, so a rule can say its result
      *replaces* the alternatives rather than joining them (Fallin, 2026). Isel
      classes carry the generic op forever at infinite cost, extraction skips it
      on every visit, and `phases::saturate_isel` exists as a separate pass only
      because extraction fails outright on a class with no machine op. Subsume
      makes that a property of the rules instead of a pass, and shrinks every
      class isel touches.
- [ ] *(unreachable: needs builtins)* **Bit instructions**: `popcnt`, `bsr`/`bsf`, `tzcnt`/`lzcnt`, `bswap`, and
      `bt` itself -- the read form, whose result is a flag and so needs the
      compare seam rather than a value class. `bt` needs a cc-carrying node like
      `X86UcomisdCc` so an `Icmp` class can take its flags from a CF-only
      instruction with the cc rewritten `Ne -> Ult`.
      **The blocker on the rest is a source idiom, not the encoding**: tinyc has
      no builtins, so nothing reaches these rules. `bswap` (a 4-way `Or` of
      masked shifts) is the only plausible one to match today.
      `bts`/`btr`/`btc` are done in both register and immediate-index forms.
- [ ] *(unreachable: needs builtins, and a CPU feature knob)* **BMI/BMI2 when available**: `andn`, `bextr`, `blsi`/`blsr`/`blsmsk`,
      `shlx`/`shrx`/`sarx` (no flag clobber, no CL constraint), `mulx`.
      Needs a CPU feature level knob that does not exist yet.
- [ ] *(unreachable: needs a source idiom)* **Carry-chain `adc`/`sbb` proper.** A multi-word add has no source idiom
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

- [ ] **Variadic function *definitions*.** The caller's half is done. The
      callee's needs one new pseudo-op -- SysV's 176-byte register save area
      has to name argument registers that are not declared parameters, which
      `Op::Param` cannot express -- and `va_arg` needs a *type* at the use
      site, so it needs builtins, which is the wall half of P2 is behind.
      Declaring one is a parse error until then.
- [ ] Switch/case + dense jump-table lowering (frontend + backend).
- [x] `>6` int / `>8` float args via stack, callee read side. A stack-passed
      double is read with `movsd`, not a `mov` of an `OpSize`, and the caller
      moves it out through the scratch GPR because `push` addresses no XMM
      register. `tests/lit/functions/stack_fp_args.c`. **Caller-side alignment
      is broken and moved to P0** -- it is a segfault, not an unblock.
- [ ] SysV struct-by-value passing/returning. Needs INTEGER/SSE/MEMORY
      classification and hidden-pointer return; the `unreachable!()` it hits
      today is listed in P0.
- [ ] Error recovery: emit a diagnostic instead of panicking on internal errors.

## Known bugs

**Nothing open.** At 400 seeds a shape `mixed`, `args` and `pressure` are all
400/400, the enumerated `abi` shape is 98/98, the saved corpus is 18 `fixed`
passing with nothing open, and `BLITZ_VERIFY` is green at `1` and `strict` across
all 566 lit tests.

The entry that used to stand here -- a register-sharing violation with no
behavioural symptom -- was a defect in the verifier's own inputs rather than in
the code it checked. See Closed.

**A green corpus is evidence about the shapes the corpus has.** The last two
wrong-value bugs were both in six-plus-argument functions that call something,
and the whole gate set -- 546 lit tests, 335 differential comparisons, 400 seeds
a shape -- was green while such a function summed its arguments to
`a+b+c+a+c+f`. One of them was *behind* a segfault: the misaligned stack killed
the program before its buffered output could disagree with anything, and the
wrong values only became visible once the crash was fixed. **Fix the crash
first, then re-read the output** -- a program that dies has not passed.

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

- **Exploration-shaped rules do not pay here, and the experiment has a number.**
  Written, measured, reverted. Two rules, both exact in wrapping arithmetic so
  neither needed the `nsw`/`nuw` flags: **distribution**
  `Mul(a, Add(b, c)) -> Add(Mul(a, b), Mul(a, c))`, the direction
  `distributive.rs` deliberately omits, and **reassociation**
  `Add(Add(a, b), c) -> Add(a, Add(b, c))`.

  | measure | result |
  | --- | --- |
  | cycles, geometric mean over 24 `live` kernels | **`+1.45%`, worse** |
  | kernels past the `~1%` noise floor, better | **1 of 24** (`funnel_mix`, `-1.2%`) |
  | kernels past it, worse | 4, two of them badly: `float_convert` `+19.1%`, `struct_fields` `+17.3%` |
  | `bench` instructions | 4 of 7 changed rows worse; `matmul` and `loop_nest` improve at `-6.7%` and `-6.4%` *copies* |
  | compile time over `bench` | 194ms to 226ms, `+16.5%` |
  | correctness | 348/348 differential with the `cc` oracle clean, so the rules are sound |

  **Two mechanisms, and the first one is the more interesting.**

  1. **They do not saturate.** Reassociating an n-term chain produces a grouping
     that is itself reassociable, so `changed` comes back true every round and the
     saturation loop runs its full 16 iterations over a graph that grows each
     time. Before the caller was changed to explore *once*, 4 of the 15 `bench`
     kernels did not finish compiling in 15 seconds. The class ceiling cannot
     brake it: `max_classes` is 500_000 and a kernel has a few hundred classes, so
     the guard is never reached while the CPU is. **So the equality-saturation
     framing does not even apply to the rules it is supposed to be for** -- the
     alternatives have to be *present*, which is one pass, not a fixpoint.
  2. **The cost model cannot price what they offer.** It prices nodes, and the
     distributed form's cost is in *register pressure* -- more values live at once
     -- which no per-node cost can see. `struct_fields` and `float_convert` are
     both cases where extraction took the distributed form and the allocator paid
     for it, `+10.1%` and `+1.2%` instructions for `+17%` and `+19%` cycles.

  **What would have to change first**: `P2`'s cost model item, since a cost that
  cannot see pressure cannot choose between these forms. Do not write more
  exploration rules until it can. The two that already pay -- commutativity and
  constant-multiply decomposition -- both offer alternatives whose cost *is*
  visible per node, which is now the distinguishing property rather than a
  coincidence.

- **A register-pressure check on the inliner is measured out, after four
  models.** The case where refusing an inline helps and the cases where it hurts
  are *indistinguishable in pressure terms*, so no arrangement of the arithmetic
  separates them. Do not attempt a fifth.

  What the four models were, and what each did:

  | model | result |
  | --- | --- |
  | charge the inlined side only | `lit` insts `-3.2%` spills `-26.1%`, `fuzz` `-1.9%` / `-24.9%`, **`bench` unchanged**, 61 rows worse |
  | require the call side to fit too | refuses nothing, zero changes anywhere |
  | compare overflow magnitudes symmetrically | `fuzz` `-4.8%` / `-30.2%`, `lit` and `bench` unchanged, 13 rows worse |
  | refuse where the enclosing loop already overflows by more than the callee adds | `live` `call_in_pressure` `-4.4%` insts / `-19.1%` reloads and `-3.7%` cycles, `bench` unchanged, **15 of 18 changed `fuzz` rows worse, to `+30.7%`** |

  The fourth is the one built on the right region: the values live across a call
  in a loop are the *header's* parameters, not the calling block's, and the
  calling block reports 5 where its loop carries 18. It gets both `live` kernels
  right -- `call_inlinable` inlines and keeps its `1.42x`, `call_in_pressure`
  refuses and takes the `3.7%`. Then `pressure-seed19` has callers overflowing by
  8 to 20 against callees bringing 4 to 6, which is the same shape as
  `call_in_pressure`'s 23 against 6, and refusing there costs `+28.7%`
  instructions.

  **Why no model can work**: the cost of refusing is mostly *lost optimization*,
  not pressure. Inlining lets constant propagation and DCE run through the call
  -- `args` seed 3 refused one inline and kept 187 instructions that would
  otherwise have folded, taking spills from 7 to 63. A pressure estimate cannot
  see that, and the two situations present identical pressure.

  **The asymmetry is the reason it is not worth more attempts.** Measured on
  `live`, inlining is worth `1.42x` where it wins and `3.8%` where it loses. A
  check that is wrong in the expensive direction once pays for every time it is
  right ten times over.

  Kept from the attempt: `tests/lit/live/call_inlinable.c` and
  `call_in_pressure.c`, which are why any of this has a cycle number --
  before them `live` had no inlinable call site at all. `src/compile/pressure.rs`
  stays for LICM, which is still budgeted by it.

  **The old criterion was also void, for an unrelated reason.** "Done when `-O1`
  beats `-O0` on all 15 `bench` kernels" became true when `-O0` moved to
  `regalloc::fast`, which puts every value in a frame slot -- `-O1` now wins by
  2 to 3x on every kernel without the inliner changing at all. A criterion
  another optimization level can satisfy was never measuring this pass.

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
- **Equality saturation converges in two rounds and is worth 0.39%.** Measured
  by setting `-O1`'s `saturation_limit` to 16, 8, 4, 2 and 1 and comparing the
  emitted code over `live` + `bench`: 16, 8, 4 and 2 are byte-identical (4390
  instructions, 15996 bytes) and 1 costs +17 instructions and +42 bytes. Nothing
  in the lit corpus hits the limit, and the loop exits on `!changed`, so 16 is a
  safety cap that costs nothing -- **do not tune it, and do not expect more
  rounds to find anything.** The rules are all "cleanup" rules, each an
  unambiguous improvement, and a rewrite that is always better does not need an
  e-graph to hold the alternative. What the e-graph is actually earning its keep
  as is the *instruction selector*: dumping multi-node classes shows they are
  projections and machine forms (`Add | Addr{scale:1} | Addr{scale:4} | X86Lea2
  | X86Lea3{scale:4}`), where the winner genuinely depends on context and the
  cost model has to choose. Average class size is 2.009 against Cranelift's
  measured 1.13, so blitz is in a wider regime than theirs -- see the experiment
  in P1 before concluding anything from that.
- **Extraction ignores shared substructure on purpose.** Optimal extraction over
  a DAG where a shared node is paid for once is NP-hard, by reduction from
  weighted set cover (Fallin, 2026). `extract` is bottom-up dynamic programming
  that costs each use independently, which is the same call Cranelift made after
  proving it. Not a gap, and not worth an ILP solver.
- **Cost-based extraction picks the instruction form.** Isel rules add every
  legal alternative to the class and let cost decide; no manual selection logic.
- **There is no `ror` or `shrd` isel form.** `ror k` is `rol (w-k)` in the same
  encoding size and `shrd y,x,w-k` computes the same bits as `shld x,y,k`, so
  the pairs differ only in which operand the destructive form consumes and
  extraction has no basis to prefer either.
