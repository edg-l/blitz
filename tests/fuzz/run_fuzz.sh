#!/bin/sh
# Generate UB-free C programs and check blitz against three oracles.
#
# The oracles themselves live in oracles.sh, shared with run_corpus.sh. This
# script is the generator driver: it decides which programs get checked.
#
# Every generated program is free of undefined behavior by construction (see
# gen_c.py), so any disagreement is a compiler bug rather than a program that
# was entitled to two answers.
#
# Usage:
#   bash tests/fuzz/run_fuzz.sh [count] [shape]
#
#   count   number of programs per shape (default 400)
#   shape   mixed | args | pressure | all  (default all)
#
# THE WIDTH IS THE POINT. A narrow sweep is not a cheaper version of this check,
# it is a different and much weaker one: at 30 seeds a shape all three shapes
# were green while seven programs miscompiled, and the -O1 allocator bug in
# corpus/fixed/args-seed310.c is at seed 310 and appears in one shape only.
# Every default here is measured rather than chosen -- 400 seeds is 51s for
# `args`, 49s for `mixed` and 64s for `pressure`, so the whole sweep is under
# three minutes, which is what the other harnesses cost.
#
# Failing programs are left in the work directory and the path is printed. Save
# one into tests/fuzz/corpus/ to make it part of the seconds-long regression
# check.
#
# Honors BLITZ_VERIFY, CC, and COMPILE_TIMEOUT (seconds per compile before it is
# reported as a hang; default 60).
#
# Set RESULTS=<path> to also write one machine-readable line per program and
# level:
#
#   <shape>-<seed> <level> pass
#   <shape>-<seed> <level> fail <kind>
#
# The shape is part of the key because one sweep covers all three and seed 1 is
# a different program in each.
#
# with kind one of no-compile, hang, exit-nonzero, wrong-predicted, wrong-cc, or
# levels-disagree (recorded against level `both`). compare_ref.sh joins two of
# these files to say which pairs a change moved, since judging a change per seed
# hides one level going backwards while the other improves.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The `checked` profile: optimized with `debug_assertions` on. A debug blitz is
# ~10x slower and the assertions are what catch a broken internal invariant, so
# neither plain profile is the right one to test against. Build it with
# `cargo build --profile checked -p tinyc -p blitztest`.
PROFILE="${PROFILE:-checked}"
TINYC="${TINYC:-$ROOT/target/$PROFILE/tinyc}"
CC="${CC:-cc}"
COUNT="${1:-400}"
SHAPE="${2:-all}"
# Seconds any single compile may take before it counts as a hang.
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-60}"

if [ ! -x "$TINYC" ]; then
    echo "error: tinyc not found at $TINYC (run 'cargo build --profile checked -p tinyc' first)" >&2
    exit 1
fi

. "$SCRIPT_DIR/oracles.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_fuzz_XXXXXX")"

if [ -n "$RESULTS" ]; then
    : > "$RESULTS"
fi

if [ "$SHAPE" = all ]; then
    SHAPES="mixed args pressure"
else
    SHAPES="$SHAPE"
fi

total_fail=0

for shape in $SHAPES; do
    pass=0
    fail=0
    skip=0

    seed=1
    while [ "$seed" -le "$COUNT" ]; do
        src="$WORK/p$shape$seed.c"
        if ! python3 "$SCRIPT_DIR/gen_c.py" --seed "$seed" --shape "$shape" --out "$src" 2>/dev/null; then
            skip=$((skip + 1))
            seed=$((seed + 1))
            continue
        fi

        st=0
        check_program "$shape-$seed" "$shape seed $seed" "$src" || st=$?
        case "$st" in
            0) pass=$((pass + 1)); printf "." ;;
            2) skip=$((skip + 1)) ;;
            *) fail=$((fail + 1)) ;;
        esac
        seed=$((seed + 1))
    done

    printf "\n\n%s programs (%s): %d passed, %d failed, %d ungeneratable\n" \
        "$COUNT" "$shape" "$pass" "$fail" "$skip"
    total_fail=$((total_fail + fail))
done

if [ "$total_fail" -eq 0 ]; then
    rm -rf "$WORK"
else
    echo "failing programs kept in $WORK"
fi
[ "$total_fail" -eq 0 ]
