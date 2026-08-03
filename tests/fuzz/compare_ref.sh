#!/bin/sh
# Compare this working tree against a git ref, per seed and optimization level.
#
# Judging a change per seed hides one level going backwards while the other
# improves, so every table here is keyed on the (seed, level) pair. It also
# reports the KIND of each failure, because a program that stops compiling and a
# program that starts printing the wrong answer are not the same news: wrong code
# outranks a refusal to compile, and a change that trades the second for the
# first is an improvement even when the pass count drops.
#
# Usage:
#   bash tests/fuzz/compare_ref.sh <ref> [count] [shape]
#
#   ref     any git revision (HEAD~1, a branch, a tag, a sha)
#   count   number of programs (default 20)
#   shape   mixed | args | pressure  (default mixed)
#
# The ref is built in its own worktree under a disk-backed cache directory, so the
# working tree is never touched and no rebuild of the current tree is forced. Both
# runs use the same seeds, and gen_c.py is deterministic per seed, so the two sides
# compile identical programs.
#
# Every ref shares ONE target directory. Cargo then rebuilds only the workspace
# crates when the ref changes, and the dependency compiles happen once ever --
# a per-ref target dir costs a full dependency build each time and defeats
# sccache too, whose cache key carries the `--extern` paths (measured: 100% hits
# with a stable target dir, 0% across two of them). Two comparisons against
# different refs at the same time will fight over it; run them one at a time.
#
# Both sides are built optimized with `debug_assertions` still on -- the same thing
# `[profile.checked]` gives, but spelled with cargo's environment overrides on top
# of `--release`, because the ref side is a checkout of an arbitrary commit and a
# commit older than that profile has no such profile to name.
#
# Optimized because this tool compiles `4 x count` programs and a debug blitz is
# ~10x slower (25.4s against 2.8s on the 4990-line `args` seed 29), which turned a
# 60-seed sweep into an hour. Assertions kept because a plain release build stops
# checking the invariants only they check.
#
# Honors CC, COMPILE_TIMEOUT and BLITZ_VERIFY (passed to both sides), and CACHE_DIR
# for where the worktree and its build go (default ~/.cache).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF="$1"
COUNT="${2:-20}"
SHAPE="${3:-mixed}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache}"

if [ -z "$REF" ]; then
    echo "usage: bash tests/fuzz/compare_ref.sh <ref> [count] [shape]" >&2
    exit 2
fi

export CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true
export CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only

SHA="$(git -C "$ROOT" rev-parse --short "$REF")"
WT="$CACHE_DIR/blitz-compare-$SHA"
WT_TARGET="$CACHE_DIR/blitz-compare-target"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/blitz_compare_XXXXXX")"

cleanup() {
    # The worktree and its target dir are kept: a second comparison against the
    # same ref then costs nothing, which is the common case while iterating.
    rm -rf "$OUT"
}
trap cleanup EXIT

echo "building $REF ($SHA) in $WT"
if [ ! -d "$WT" ]; then
    git -C "$ROOT" worktree add --detach -q "$WT" "$SHA"
fi
( cd "$WT" && CARGO_TARGET_DIR="$WT_TARGET" cargo build -q --release -p tinyc )

echo "building the working tree"
( cd "$ROOT" && cargo build -q --release -p tinyc )

echo "running $COUNT $SHAPE programs on each side"
RESULTS="$OUT/ref.txt" TINYC="$WT_TARGET/release/tinyc" \
    sh "$SCRIPT_DIR/run_fuzz.sh" "$COUNT" "$SHAPE" > "$OUT/ref.log" 2>&1 || true
RESULTS="$OUT/now.txt" TINYC="$ROOT/target/release/tinyc" \
    sh "$SCRIPT_DIR/run_fuzz.sh" "$COUNT" "$SHAPE" > "$OUT/now.log" 2>&1 || true

# One row per (seed, level) that changed, plus the totals. `join` needs the pair
# as a single key, so the seed and level are glued with a colon.
key() {
    awk '{ printf "%s:%s %s\n", $1, $2, ($3 == "pass" ? "pass" : "fail " $4) }' "$1" \
        | sort -k1,1
}
key "$OUT/ref.txt" > "$OUT/ref.keyed"
key "$OUT/now.txt" > "$OUT/now.keyed"

printf '\n'
# Rows are accumulated in input order rather than iterated out of an associative
# array, so two runs of the same comparison print the same thing.
join -a1 -a2 -e MISSING -o 0,1.2,1.3,2.2,2.3 "$OUT/ref.keyed" "$OUT/now.keyed" \
    | awk '
    {
        pair = $1
        was = $2 == "fail" ? $2 " " $3 : $2
        now = $4 == "fail" ? $4 " " $5 : $4
        if (was == now) { same++; next }
        if (was == "pass" && now ~ /^fail/) { reg[++nreg] = sprintf("  %-12s now %s", pair, now) }
        else if (was ~ /^fail/ && now == "pass") { fix[++nfix] = sprintf("  %-12s was %s", pair, was) }
        else { chg[++nchg] = sprintf("  %-12s %s -> %s", pair, was, now) }
    }
    END {
        if (nreg) { print "REGRESSED (was passing):"; for (i = 1; i <= nreg; i++) print reg[i] }
        if (nfix) { print "FIXED (was failing):"; for (i = 1; i <= nfix; i++) print fix[i] }
        if (nchg) {
            print "FAILURE KIND CHANGED (both sides fail):"
            for (i = 1; i <= nchg; i++) print chg[i]
        }
        printf "\n%d pair(s) unchanged, %d regressed, %d fixed, %d changed kind\n",
            same, nreg, nfix, nchg
    }'

printf '\nref: %s   now: working tree\n' "$SHA"
tail -3 "$OUT/ref.log" | grep programs | sed 's/^/  ref: /' || true
tail -3 "$OUT/now.log" | grep programs | sed 's/^/  now: /' || true
