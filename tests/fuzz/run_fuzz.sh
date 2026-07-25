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
# Honors BLITZ_VERIFY and CC.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TINYC="${TINYC:-$ROOT/target/debug/tinyc}"
CC="${CC:-cc}"
COUNT="${1:-20}"
SHAPE="${2:-mixed}"

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

    if ! "$TINYC" "$src" -O1 -o "$WORK/o1" > "$WORK/log" 2>&1; then
        fail=$((fail + 1))
        printf "\nFAIL seed %s: blitz -O1 did not compile\n  %s\n" "$seed" "$src"
        head -2 "$WORK/log" | sed 's/^/  /'
        seed=$((seed + 1))
        continue
    fi
    if ! "$TINYC" "$src" -O0 -o "$WORK/o0" > "$WORK/log" 2>&1; then
        fail=$((fail + 1))
        printf "\nFAIL seed %s: blitz -O0 did not compile\n  %s\n" "$seed" "$src"
        head -2 "$WORK/log" | sed 's/^/  /'
        seed=$((seed + 1))
        continue
    fi

    got1="$(timeout 20 "$WORK/o1" 2>/dev/null)"; st1=$?
    got0="$(timeout 20 "$WORK/o0" 2>/dev/null)"; st0=$?

    if [ "$st1" -ne 0 ] || [ "$st0" -ne 0 ]; then
        fail=$((fail + 1))
        printf "\nFAIL seed %s: nonzero exit (-O0 %s, -O1 %s)\n  %s\n" \
            "$seed" "$st0" "$st1" "$src"
        seed=$((seed + 1))
        continue
    fi
    if [ "$got0" != "$got1" ]; then
        fail=$((fail + 1))
        printf "\nFAIL seed %s: -O0 printed %s, -O1 printed %s\n  %s\n" \
            "$seed" "$got0" "$got1" "$src"
        seed=$((seed + 1))
        continue
    fi
    if [ "$got1" != "$want" ]; then
        fail=$((fail + 1))
        printf "\nFAIL seed %s: blitz printed %s, generator predicted %s\n  %s\n" \
            "$seed" "$got1" "$want" "$src"
        seed=$((seed + 1))
        continue
    fi

    if command -v "$CC" > /dev/null 2>&1; then
        if "$CC" -w -O0 -ffp-contract=off -x c "$src" -o "$WORK/ref" 2>/dev/null; then
            gotc="$(timeout 20 "$WORK/ref" 2>/dev/null)"
            if [ "$gotc" != "$got1" ]; then
                fail=$((fail + 1))
                printf "\nFAIL seed %s: blitz printed %s, %s printed %s\n  %s\n" \
                    "$seed" "$got1" "$CC" "$gotc" "$src"
                seed=$((seed + 1))
                continue
            fi
        fi
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
