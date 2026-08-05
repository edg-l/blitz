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
#
#   CORPUS=lit|bench|fuzz|all   which programs (default all)
#   SEEDS=30                    generated programs per shape, for the fuzz corpus
#   JOBS=N                      parallel compiles (default: every core)
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
CORPUS="${CORPUS:-all}"
SEEDS="${SEEDS:-30}"

# One worker run: compile one program at one level and print its row.
if [ "$1" = "--one" ]; then
    name="$2"
    file="$3"
    level="$4"
    out="$(BLITZ_DEBUG=stats "$TINYC" "$level" "$file" -o "$5/a.out" 2>&1 >/dev/null || true)"
    # A program that does not compile is a hole in the table, kept visible.
    if ! printf '%s' "$out" | grep -q 'blitz::stats'; then
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

mode=print
case "$1" in
    --check) mode=check ;;
    --update) mode=update ;;
    "") ;;
    *) echo "usage: bash tests/run_codesize.sh [--check|--update]" >&2; exit 2 ;;
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
    export SELF TINYC
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

case "$CORPUS" in
    all) corpora="lit bench fuzz" ;;
    *) corpora="$CORPUS" ;;
esac

regressed=0
for corpus in $corpora; do
    measure "$corpus"
    case "$mode" in
        print)
            printf '# %s\n' "$corpus"
            cat "$WORK/$corpus.tsv"
            ;;
        update)
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
