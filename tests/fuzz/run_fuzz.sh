#!/bin/sh
# Generate UB-free C programs and check blitz against three oracles.
#
#   predicted   the generator interprets the program as it builds it, so it
#               knows the answer before any compiler runs
#   -O0 vs -O1  self-consistency: optimization must not change behavior
#   vs cc       ground truth, catching bugs that are equally wrong at both
#               optimization levels
#
# Every generated program is free of undefined behavior by construction (see
# gen_c.py), so any disagreement is a compiler bug rather than a program that
# was entitled to two answers.
#
# Usage:
#   bash tests/fuzz/run_fuzz.sh [count] [shape]
#
#   count   number of programs (default 20)
#   shape   mixed | args | pressure  (default mixed)
#
# Failing programs are left in the work directory and the path is printed.
# Honors BLITZ_VERIFY, CC, and COMPILE_TIMEOUT (seconds per compile before it is
# reported as a hang; default 60).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TINYC="${TINYC:-$ROOT/target/debug/tinyc}"
CC="${CC:-cc}"
COUNT="${1:-20}"
SHAPE="${2:-mixed}"
# Seconds any single compile may take before it counts as a hang.
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-60}"

if [ ! -x "$TINYC" ]; then
    echo "error: tinyc not found at $TINYC (run 'cargo build -p tinyc' first)" >&2
    exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_fuzz_XXXXXX")"

pass=0
fail=0
skip=0

seed=1
while [ "$seed" -le "$COUNT" ]; do
    src="$WORK/p$seed.c"
    if ! python3 "$SCRIPT_DIR/gen_c.py" --seed "$seed" --shape "$SHAPE" --out "$src" 2>/dev/null; then
        skip=$((skip + 1))
        seed=$((seed + 1))
        continue
    fi

    want="$(sed -n 's|^// OUTPUT: ||p' "$src")"

    # A reference answer, when the reference compiler will take the program.
    wantc=""
    if command -v "$CC" > /dev/null 2>&1 \
        && "$CC" -w -O0 -ffp-contract=off -x c "$src" -o "$WORK/ref" 2>/dev/null; then
        wantc="$(timeout 20 "$WORK/ref" 2>/dev/null)" || wantc=""
    fi

    # Check each level on its own, and keep going after a failure.
    #
    # These used to short-circuit: -O1 was compiled first and a failure there
    # skipped the seed entirely. A level that cannot allocate registers then hid
    # whatever the other level did, and three -O0 miscompiles sat behind -O1
    # compile errors for a whole session. A compile error at one level says
    # nothing about the other.
    seed_failed=0
    o0_out=""; o0_ok=0
    o1_out=""; o1_ok=0
    for level in -O0 -O1; do
        # Compilation is under a timeout of its own. A compiler that loops
        # forever otherwise absorbs the entire run: one hang in the parallel-copy
        # sequentializer ate a 40-program sweep before anyone noticed the harness
        # was not merely slow. A hang is a finding, and reported as one.
        if ! timeout "$COMPILE_TIMEOUT" "$TINYC" "$src" "$level" -o "$WORK/o" \
            > "$WORK/log" 2>&1; then
            st=$?
            seed_failed=1
            if [ "$st" -eq 124 ]; then
                printf "\nFAIL seed %s: blitz %s HUNG (over %ss)\n  %s\n" \
                    "$seed" "$level" "$COMPILE_TIMEOUT" "$src"
            else
                printf "\nFAIL seed %s: blitz %s did not compile\n  %s\n" "$seed" "$level" "$src"
                head -2 "$WORK/log" | sed 's/^/  /'
            fi
            continue
        fi
        # `set -e` is on, and a program under test may legitimately exit
        # nonzero -- that is a finding, not a reason to abandon the run. The
        # original code had the same shape and aborted the whole harness the
        # first time a compiled program returned nonzero.
        if out="$(timeout 20 "$WORK/o" 2>/dev/null)"; then st=0; else st=$?; fi
        if [ "$level" = "-O0" ]; then o0_out="$out"; o0_ok=1; else o1_out="$out"; o1_ok=1; fi
        if [ "$st" -ne 0 ]; then
            seed_failed=1
            printf "\nFAIL seed %s: blitz %s exited %s\n  %s\n" "$seed" "$level" "$st" "$src"
            continue
        fi
        if [ -n "$want" ] && [ "$out" != "$want" ]; then
            seed_failed=1
            printf "\nFAIL seed %s: blitz %s printed %s, generator predicted %s\n  %s\n" \
                "$seed" "$level" "$out" "$want" "$src"
            continue
        fi
        if [ -n "$wantc" ] && [ "$out" != "$wantc" ]; then
            seed_failed=1
            printf "\nFAIL seed %s: blitz %s printed %s, %s printed %s\n  %s\n" \
                "$seed" "$level" "$out" "$CC" "$wantc" "$src"
        fi
    done

    # -O0-vs-O1 self-consistency, when both levels produced a program. This
    # catches a pass that changes behaviour even where no oracle disagrees.
    if [ "$o0_ok" = 1 ] && [ "$o1_ok" = 1 ] && [ "$o0_out" != "$o1_out" ]; then
        seed_failed=1
        printf "\nFAIL seed %s: -O0 printed %s, -O1 printed %s\n  %s\n" \
            "$seed" "$o0_out" "$o1_out" "$src"
    fi

    if [ "$seed_failed" = 1 ]; then
        fail=$((fail + 1))
        seed=$((seed + 1))
        continue
    fi

    pass=$((pass + 1))
    printf "."
    seed=$((seed + 1))
done

printf "\n\n%s programs (%s): %d passed, %d failed, %d ungeneratable\n" \
    "$COUNT" "$SHAPE" "$pass" "$fail" "$skip"

if [ "$fail" -eq 0 ]; then
    rm -rf "$WORK"
else
    echo "failing programs kept in $WORK"
fi
[ "$fail" -eq 0 ]
