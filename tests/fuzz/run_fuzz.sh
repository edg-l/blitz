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
#   count   number of programs (default 20)
#   shape   mixed | args | pressure  (default mixed)
#
# Failing programs are left in the work directory and the path is printed. Save
# one into tests/fuzz/corpus/ to make it part of the seconds-long regression
# check; a 200-seed sweep is minutes and is not run between every change.
#
# Honors BLITZ_VERIFY, CC, and COMPILE_TIMEOUT (seconds per compile before it is
# reported as a hang; default 60).
#
# Set RESULTS=<path> to also write one machine-readable line per seed and level:
#
#   <seed> <level> pass
#   <seed> <level> fail <kind>
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
COUNT="${1:-20}"
SHAPE="${2:-mixed}"
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

    st=0
    check_program "$seed" "seed $seed" "$src" || st=$?
    case "$st" in
        0) pass=$((pass + 1)); printf "." ;;
        2) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
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
