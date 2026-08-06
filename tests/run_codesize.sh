#!/bin/sh
# Did a change make the emitted code better or worse?
#
# For every (program, level) pair this records four numbers about the code
# blitz emits -- instructions, `.text` bytes, spill stores, reloads -- and
# compares them against checked-in baselines. It answers the question no other
# harness here answers: `run_identity.sh` says whether the output changed at
# all, `run_diff.sh` says whether it is correct, and neither says whether it got
# better.
#
# The numbers come from the compiler, not from a disassembly: `BLITZ_DEBUG=stats`
# prints one line per function with the counts taken off the final instruction
# stream, so a spill is what the processor executes rather than what a pass
# planned. Rows are per program, summed over its functions.
#
# Usage:
#   bash tests/run_codesize.sh                 # print the current table
#   bash tests/run_codesize.sh --check         # compare against the baselines
#   bash tests/run_codesize.sh --update        # rewrite the baselines
#   bash tests/run_codesize.sh --gap           # how far the output is from gcc/clang -O2
#
#   CORPUS=lit|bench|fuzz|all   which programs (default all; --gap defaults to bench)
#   SEEDS=30                    generated programs per shape, for the fuzz corpus
#   JOBS=N                      parallel compiles (default: every core)
#   GAP_LEVEL=-O1               the blitz level `--gap` compares (default -O1)
#   GAP_CC="gcc clang"          the reference compilers `--gap` measures
#
# `--gap` answers a different question from the other three modes, and it is the
# only one that measures the project's actual goal. The baselines say whether
# this compiler is better than it was last week; they cannot say whether it is
# any good, because they compare it only against itself. `--gap` compiles the
# same program with `gcc -O2` and `clang -O2` and reports the ratio.
#
# All three compilers are measured THE SAME WAY -- disassemble the object,
# count instructions, sum their encoded bytes -- rather than taking blitz's
# number from `BLITZ_DEBUG=stats` and the others from a disassembly. Two
# counting methods produce two numbers whose difference is partly the method,
# and a ratio built out of that is not a measurement. Alignment padding between
# functions is excluded for the same reason: gcc emits it and blitz does not, so
# counting it would price a linker convention as code quality.
#
# WHAT A RATIO BELOW 1.0 MEANS, and it is not "blitz won". This counts static
# instructions, which is a proxy for code quality and not the thing itself. A
# reference compiler that unrolls a loop, inlines a callee into three call sites
# or vectorizes a reduction emits MORE instructions and runs FASTER. So a
# program where blitz is under 1.0 is a program to go and read, not a win to
# bank: the likely finding is a transform blitz does not do. Reading the ratio
# as a score is the one way this harness can mislead.
#
# `--gap` is not a gate and writes no baseline. A ratio moves when gcc changes
# version, so freezing one would record this machine's toolchain as a
# requirement.
#
# `--check` exits non-zero when any number went up, and prints every change
# either way. A number going up is not automatically a bug -- inlining more
# trades instructions for calls -- but it is always something to have decided
# rather than discovered later.
#
# THE THREE CORPORA ANSWER DIFFERENT QUESTIONS, and the split is deliberate:
#
#   lit    337 checked-in programs, mean 28 lines. Complete and stable: every
#          one compiles at both levels, so this table is never partial. Too
#          small to show much on its own -- prologue and call sequence dominate
#          -- but it catches a regression anywhere in the language.
#   bench  the kernels in tests/lit/bench: sieve, matmul, sorts, CRC, string
#          ops, a struct-field walk, FP loops. This is where loop-invariant
#          motion, strength reduction and alias analysis have room to show.
#   fuzz   generated programs at real register pressure, the same ones
#          `compare_ref.sh` judges for correctness. Partial by construction:
#          a program that does not compile is recorded as a hole, not skipped,
#          because a baseline that quietly omits its hard cases reads as
#          coverage it does not have.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_DIR="$ROOT/tests/baselines"
PROFILE="${PROFILE:-checked}"
TINYC="${TINYC:-$ROOT/target/$PROFILE/tinyc}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
LEVELS="${LEVELS:--O0 -O1}"
# Left empty when unset so the default can depend on the mode; see the dispatch.
CORPUS="${CORPUS:-}"
SEEDS="${SEEDS:-30}"
# Extra tinyc flags, for attributing a number to a pass: run the table again
# with `BLITZ_PASSES=-licm` and the difference is what that pass cost or
# bought. Not part of a baseline -- `--update` with FLAGS set would record a
# pipeline nobody compiles with.
FLAGS="${FLAGS:-}"
GAP_LEVEL="${GAP_LEVEL:--O1}"
GAP_CC="${GAP_CC:-gcc clang}"

# Instructions, encoded bytes and backward branches in an object file: one line
# `<insts> <bytes> <loops>`, or `- - -` if the object cannot be read.
#
# Alignment padding is excluded -- `nop` in every encoding it has, and the
# `xchg ax,ax` that is one of them, is what a linker wants rather than what a
# program executes.
#
# The backward branches are how `--gap` knows the two compilers were given the
# same problem. See `report_gap` below.
count_object() {
    objdump -d "$1" 2>/dev/null | awk -F'\t' '
        function nbytes(s,   arr, k, i, c) {
            k = split(s, arr, / +/)
            for (i = 1; i <= k; i++) if (arr[i] ~ /^[0-9a-f][0-9a-f]$/) c++
            return c
        }
        function hex2dec(s,   i, c, v, d) {
            v = 0; s = tolower(s)
            for (i = 1; i <= length(s); i++) {
                d = index("0123456789abcdef", substr(s, i, 1)) - 1
                if (d < 0) break
                v = v * 16 + d
            }
            return v
        }
        /^[ ]*[0-9a-f]+:/ {
            # A long instruction wraps onto continuation lines that carry bytes
            # and no mnemonic; they belong to whatever came before them.
            if (NF < 3 || $3 == "") {
                if (!padding) bytes += nbytes($2)
                next
            }
            addr = $1; sub(/:.*$/, "", addr); gsub(/ /, "", addr)
            insn = $3; sub(/^ +/, "", insn)
            padding = (insn ~ /^(nop|xchg +ax,ax|data16|cs )/)
            if (padding) next
            insts++
            bytes += nbytes($2)
            if (insn ~ /^j/) {
                target = insn
                sub(/^[a-z0-9]+ +/, "", target)
                sub(/[^0-9a-fA-F].*$/, "", target)
                if (target != "" && hex2dec(target) < hex2dec(addr)) loops++
            }
        }
        END {
            if (insts > 0) printf "%d %d %d\n", insts, bytes, loops
            else print "- - -"
        }
    '
}

# One worker run: compile one program at one level and print its row.
if [ "$1" = "--one" ]; then
    name="$2"
    file="$3"
    level="$4"
    # shellcheck disable=SC2086 -- FLAGS is a list of flags on purpose.
    out="$(BLITZ_DEBUG=stats "$TINYC" "$level" $FLAGS "$file" -o "$5/a.out" 2>&1 >/dev/null || true)"
    # A program the compiler could not compile is a hole in the table, kept
    # visible.
    #
    # A compiler diagnostic decides it, not the exit status and not the presence
    # of stats lines. Neither of those alone is right: `stats` prints per
    # function as each is emitted, so a compile failing on the last function has
    # already printed rows for the others, and summing those recorded a partial
    # program as a real one -- the row then moved when the failing function
    # started compiling, which read as a twelvefold regression when what had
    # happened was a function appearing for the first time. The exit status is
    # not right either: the multifile tests and `dce/no_fold_dynamic.c` compile
    # completely and fail at the *link*, on a symbol another file defines, and
    # their code size is a real measurement.
    if printf '%s' "$out" | grep -q "phase '" ||
        ! printf '%s' "$out" | grep -q 'blitz::stats'; then
        printf '%s\t%s\t-\t-\t-\t-\n' "$name" "${level#-}"
        exit 0
    fi
    printf '%s' "$out" | awk -v name="$name" -v level="${level#-}" '
        /blitz::stats/ {
            for (i = 1; i <= NF; i++) {
                split($i, kv, "=")
                if (kv[1] == "insts")   insts   += kv[2]
                if (kv[1] == "bytes")   bytes   += kv[2]
                if (kv[1] == "spills")  spills  += kv[2]
                if (kv[1] == "reloads") reloads += kv[2]
            }
        }
        END { printf "%s\t%s\t%d\t%d\t%d\t%d\n", name, level, insts, bytes, spills, reloads }
    '
    exit 0
fi

# One worker run for `--gap`: compile one program with blitz and with each
# reference compiler, and count all of them off their objects.
if [ "$1" = "--gapone" ]; then
    name="$2"
    file="$3"
    dir="$4"
    row="$name"
    # shellcheck disable=SC2086 -- FLAGS is a list of flags on purpose.
    if "$TINYC" "$GAP_LEVEL" $FLAGS -c "$file" -o "$dir/blitz.o" >/dev/null 2>&1; then
        row="$row	$(count_object "$dir/blitz.o" | tr ' ' '\t')"
    else
        row="$row	-	-	-"
    fi
    for cc in $GAP_CC; do
        # `-w`: the corpus declares printf itself, which every reference
        # compiler warns about and none of them refuses.
        if "$cc" -O2 -w -c "$file" -o "$dir/$cc.o" >/dev/null 2>&1; then
            row="$row	$(count_object "$dir/$cc.o" | tr ' ' '\t')"
        else
            row="$row	-	-	-"
        fi
    done
    printf '%s\n' "$row"
    exit 0
fi

mode=print
case "$1" in
    --check) mode=check ;;
    --update) mode=update ;;
    --gap) mode=gap ;;
    "") ;;
    *) echo "usage: bash tests/run_codesize.sh [--check|--update|--gap]" >&2; exit 2 ;;
esac

if [ ! -x "$TINYC" ]; then
    echo "no tinyc at $TINYC" >&2
    echo "build it with: cargo build --profile $PROFILE -p tinyc" >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_codesize_XXXXXX")"
trap 'rm -rf "$WORK"' EXIT INT TERM
mkdir -p "$BASELINE_DIR"

# Collect the programs of one corpus as `<name>|<path>` lines.
collect() {
    case "$1" in
        lit)
            find "$ROOT/tests/lit" -name '*.c' -not -path '*/bench/*' | sort |
                while IFS= read -r f; do
                    printf '%s|%s\n' "${f#"$ROOT"/tests/lit/}" "$f"
                done
            ;;
        bench)
            find "$ROOT/tests/lit/bench" -name '*.c' | sort |
                while IFS= read -r f; do
                    printf '%s|%s\n' "${f#"$ROOT"/tests/lit/bench/}" "$f"
                done
            ;;
        fuzz)
            # Regenerated from seeds rather than checked in: gen_c.py is
            # deterministic per (seed, shape), so the programs are reproducible
            # without carrying thousands of lines in the tree.
            mkdir -p "$WORK/fuzz"
            for shape in mixed args pressure; do
                i=0
                while [ "$i" -lt "$SEEDS" ]; do
                    f="$WORK/fuzz/$shape-$i.c"
                    if python3 "$ROOT/tests/fuzz/gen_c.py" --seed "$i" --shape "$shape" \
                        --out "$f" >/dev/null 2>&1; then
                        printf '%s|%s\n' "$shape-seed$i" "$f"
                    fi
                    i=$((i + 1))
                done
            done
            ;;
    esac
}

# Build the table for one corpus into $WORK/<corpus>.tsv.
measure() {
    corpus="$1"
    collect "$corpus" > "$WORK/files.$corpus"
    : > "$WORK/jobs.$corpus"
    n=0
    while IFS='|' read -r name file; do
        for level in $LEVELS; do
            n=$((n + 1))
            printf '%s\t%s\t%s\t%s\n' "$name" "$file" "$level" "$WORK/j$n" >> "$WORK/jobs.$corpus"
            mkdir -p "$WORK/j$n"
        done
    done < "$WORK/files.$corpus"

    SELF="$SCRIPT_DIR/run_codesize.sh"
    export SELF TINYC FLAGS
    # `|| true`: a worker that fails still wrote its row, and xargs exiting
    # non-zero under `set -e` would end the run before the rows are read.
    xargs -d '\n' -P "$JOBS" -n1 sh -c '
        name=$(printf "%s" "$1" | cut -f1)
        file=$(printf "%s" "$1" | cut -f2)
        level=$(printf "%s" "$1" | cut -f3)
        dir=$(printf "%s" "$1" | cut -f4)
        sh "$SELF" --one "$name" "$file" "$level" "$dir"
    ' _ < "$WORK/jobs.$corpus" 2>/dev/null | sort > "$WORK/$corpus.tsv" || true
}

# Build the gap table for one corpus into $WORK/gap.<corpus>.tsv. One row per
# program rather than per (program, level): the question is how the code this
# compiler ships compares, and it ships one level.
measure_gap() {
    corpus="$1"
    collect "$corpus" > "$WORK/files.$corpus"
    : > "$WORK/gapjobs.$corpus"
    n=0
    while IFS='|' read -r name file; do
        n=$((n + 1))
        printf '%s\t%s\t%s\n' "$name" "$file" "$WORK/g$n" >> "$WORK/gapjobs.$corpus"
        mkdir -p "$WORK/g$n"
    done < "$WORK/files.$corpus"

    SELF="$SCRIPT_DIR/run_codesize.sh"
    export SELF TINYC FLAGS GAP_LEVEL GAP_CC
    xargs -d '\n' -P "$JOBS" -n1 sh -c '
        name=$(printf "%s" "$1" | cut -f1)
        file=$(printf "%s" "$1" | cut -f2)
        dir=$(printf "%s" "$1" | cut -f3)
        sh "$SELF" --gapone "$name" "$file" "$dir"
    ' _ < "$WORK/gapjobs.$corpus" 2>/dev/null | sort > "$WORK/gap.$corpus.tsv" || true
}

# Report one corpus's distance from the reference compilers.
#
# FOLDED PROGRAMS ARE EXCLUDED, and this is the whole reason the harness counts
# backward branches. A program that reads nothing at runtime has one answer, and
# `gcc -O2` evaluates the entire thing at compile time and emits the constant:
# the generated corpus compiles to `mov $0x562,%esi; call printf`. Comparing
# against that measures whether the reference has interprocedural constant
# propagation, not how good this compiler's code is -- and it does not measure
# it subtly, it reports a 47x gap. So a program where blitz emits a loop and the
# reference emits none is counted and named, never averaged in.
#
# The headline is a GEOMETRIC mean of the per-program ratios. An arithmetic mean
# of ratios is dominated by whichever program happens to be largest, and "twice
# as many instructions" and "half as many" have to weigh the same or the number
# says nothing about a compiler.
report_gap() {
    corpus="$1"
    awk -F'\t' -v corpus="$corpus" -v ccs="$GAP_CC" -v level="$GAP_LEVEL" '
        BEGIN { ncc = split(ccs, cc, / +/) }
        NF >= 4 {
            rows++
            if ($2 == "-") { nocompile++; next }
            line = sprintf("  %-42s %6d insts %8d bytes", $1, $2, $3)
            shown = 0
            for (i = 1; i <= ncc; i++) {
                ins = $(2 + 3 * i); loops = $(4 + 3 * i)
                if (ins == "-" || ins + 0 == 0) { line = line sprintf("   %s -", cc[i]); continue }
                if ($4 + 0 > 0 && loops + 0 == 0) {
                    line = line sprintf("   %s folded", cc[i]); folded[i]++
                    continue
                }
                r = $2 / ins
                line = line sprintf("   %s x%.2f", cc[i], r)
                shown = 1
                logsum[i] += log(r); n[i]++
                mine[i] += $2; theirs[i] += ins
                if (r > worst[i]) { worst[i] = r; worstname[i] = $1 }
            }
            if (shown) print line
        }
        END {
            printf "%s: %d programs", corpus, rows
            if (nocompile) printf ", %d blitz did not compile", nocompile
            printf "\n"
            for (i = 1; i <= ncc; i++) {
                if (n[i] == 0) {
                    printf "  vs %s -O2: nothing comparable (%d of %d folded to a constant)\n",
                        cc[i], folded[i], rows
                    continue
                }
                printf "  blitz %s vs %s -O2: x%.2f geomean over %d comparable" \
                       " (totals %d vs %d insts, x%.2f); worst x%.2f on %s",
                    level, cc[i], exp(logsum[i] / n[i]), n[i],
                    mine[i], theirs[i], mine[i] / theirs[i], worst[i], worstname[i]
                if (folded[i]) printf "; %d folded to a constant, excluded", folded[i]
                printf "\n"
            }
        }
    ' "$WORK/gap.$corpus.tsv"
}

# Compare one corpus against its baseline. Prints changes; sets `regressed`.
compare() {
    corpus="$1"
    baseline="$BASELINE_DIR/codesize-$corpus.tsv"
    if [ ! -f "$baseline" ]; then
        echo "$corpus: no baseline yet (run --update)"
        return 0
    fi
    awk -F'\t' -v corpus="$corpus" '
        FNR == NR {
            if ($1 ~ /^#/ || NF < 6) next
            key = $1 "\t" $2
            old[key] = $3 "\t" $4 "\t" $5 "\t" $6
            next
        }
        {
            if (NF < 6) next
            key = $1 "\t" $2
            new = $3 "\t" $4 "\t" $5 "\t" $6
            if (!(key in old)) { added++; printf "  + %-44s %s\n", key, "new program"; next }
            if (old[key] == new) { same++; next }
            split(old[key], o, "\t")
            split($0, n, "\t")
            changed++
            label[1] = "insts"; label[2] = "bytes"; label[3] = "spills"; label[4] = "reloads"
            line = ""
            worse = 0
            for (i = 1; i <= 4; i++) {
                ov = o[i]; nv = n[i + 2]
                if (ov == nv) continue
                if (ov == "-" || nv == "-") {
                    line = line sprintf("  %s %s -> %s", label[i], ov, nv)
                    if (nv == "-") worse = 1
                    continue
                }
                pct = (ov + 0 == 0) ? 0 : (nv - ov) * 100.0 / ov
                line = line sprintf("  %s %d -> %d (%+.1f%%)", label[i], ov, nv, pct)
                if (nv + 0 > ov + 0) worse = 1
            }
            if (worse) { regressed++; line = line "  REGRESSION" }
            printf "  %-44s%s\n", key, line
        }
        END {
            printf "%s: %d unchanged, %d changed, %d new, %d regressed\n",
                corpus, same, changed, added, regressed
            if (regressed > 0) exit 3
        }
    ' "$baseline" "$WORK/$corpus.tsv" || regressed=1
}

# `--gap` defaults to `bench` alone, and this is a limit of the corpora rather
# than of the harness. `lit` and `fuzz` are closed-form: nothing they compute
# depends on anything read at runtime, so a reference compiler evaluates the
# program and prints the answer. The loop test below catches that when it is
# total, but not when it is partial -- a generated program's `main` folds to one
# constant while its helpers survive, because a non-static function has to be
# emitted whether or not any call to it is left. There is no detector for that;
# there is only a corpus whose inputs are not known until it runs. `bench` is
# that corpus, which is why it exists, and every one of its kernels compares.
if [ -z "$CORPUS" ]; then
    case "$mode" in
        gap) CORPUS=bench ;;
        *) CORPUS=all ;;
    esac
fi
case "$CORPUS" in
    all) corpora="lit bench fuzz" ;;
    *) corpora="$CORPUS" ;;
esac

if [ "$mode" = gap ]; then
    for cc in $GAP_CC; do
        if ! command -v "$cc" > /dev/null 2>&1; then
            echo "no $cc on PATH; set GAP_CC to the compilers you have" >&2
            exit 2
        fi
    done
    if ! command -v objdump > /dev/null 2>&1; then
        echo "no objdump on PATH" >&2
        exit 2
    fi
    for corpus in $corpora; do
        measure_gap "$corpus"
        report_gap "$corpus"
    done
    exit 0
fi

regressed=0
for corpus in $corpora; do
    measure "$corpus"
    case "$mode" in
        print)
            printf '# %s\n' "$corpus"
            cat "$WORK/$corpus.tsv"
            ;;
        update)
            if [ -n "$FLAGS" ]; then
                echo "refusing to write a baseline with FLAGS set: $FLAGS" >&2
                exit 2
            fi
            {
                printf '# program\tlevel\tinsts\tbytes\tspills\treloads\n'
                printf '# corpus: %s -- written by tests/run_codesize.sh --update\n' "$corpus"
                cat "$WORK/$corpus.tsv"
            } > "$BASELINE_DIR/codesize-$corpus.tsv"
            echo "$corpus: wrote $(grep -c . < "$WORK/$corpus.tsv") rows"
            ;;
        check) compare "$corpus" ;;
    esac
done

[ "$regressed" -eq 0 ]
