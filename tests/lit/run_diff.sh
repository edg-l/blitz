#!/bin/sh
# Differential execution harness. For every runnable lit test it compiles at
# -O0 and -O1, and with a reference compiler, then compares exit status and
# stdout across all three.
#
# Two oracles, because they fail differently:
#
#   -O0 vs -O1   Self-consistency. Needs no expected output, only that
#                optimization not change observable behavior. Catches any pass
#                that miscompiles. Blind to a bug that is equally wrong at both
#                levels.
#
#   vs cc        Ground truth. Catches the bugs self-consistency cannot see.
#                Both the cvtsi2sd REX.W bug and the missing variadic AL were
#                invisible to the -O0/-O1 comparison and only showed up here.
#
# A mismatch on either is a wrong-code bug, and the reduced case is already a
# .c file that belongs in tests/lit/.
#
# The regular suite (run_tests.sh) checks each file against its own directives
# at whatever flags its RUN line names, which for most files is one level only.
# This covers the rest.
#
# Usage:
#   bash tests/lit/run_diff.sh              # every runnable test
#   bash tests/lit/run_diff.sh arithmetic   # only paths matching a substring
#   NO_ORACLE=1 bash tests/lit/run_diff.sh  # skip the reference-compiler leg
#   CC=clang bash tests/lit/run_diff.sh     # use a different reference
#
# Honors BLITZ_VERIFY, so `BLITZ_VERIFY=strict bash tests/lit/run_diff.sh`
# checks IR invariants on every compilation.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TINYC="${TINYC:-$ROOT/target/debug/tinyc}"
CC="${CC:-cc}"
FILTER="${1:-}"

# The reference leg is optional: it needs a working cc, and a test whose C the
# reference rejects is skipped rather than failed.
use_oracle=1
if [ -n "${NO_ORACLE:-}" ] || ! command -v "$CC" > /dev/null 2>&1; then
    use_oracle=0
fi

if [ ! -x "$TINYC" ]; then
    echo "error: tinyc not found at $TINYC (run 'cargo build -p tinyc' first)" >&2
    exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_diff_XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

total=0
passed=0
failed=0
skipped=0
oracle_checked=0
oracle_failed=0
oracle_skipped=0

# Compile $1 at optimization level $2 into $WORK, run it, and write the exit
# status to $WORK/status.$2 and stdout to $WORK/out.$2.
# Returns 1 if compilation itself failed.
build_and_run() {
    _file="$1"
    _level="$2"
    _extras="$3"
    _bin="$WORK/bin.$_level"

    if ! "$TINYC" "$_file" $_extras "$_level" -o "$_bin" > "$WORK/cc.$_level" 2>&1; then
        return 1
    fi
    timeout 10 "$_bin" > "$WORK/out.$_level" 2>/dev/null
    echo $? > "$WORK/status.$_level"
    return 0
}

# Compile $1 (plus $2 extra sources) with the reference compiler and run it,
# writing status to $WORK/status.cc and stdout to $WORK/out.cc.
#
# -w because lit tests redeclare library functions with tinyc-compatible
# prototypes (`extern int printf(char* fmt, double x);`), which the reference
# warns about but compiles correctly.
build_and_run_cc() {
    _file="$1"
    _extras="$2"
    if ! "$CC" -w -O0 -x c "$_file" $_extras -o "$WORK/bin.cc" > "$WORK/cc.ref" 2>&1; then
        return 1
    fi
    timeout 10 "$WORK/bin.cc" > "$WORK/out.cc" 2>/dev/null
    echo $? > "$WORK/status.cc"
    return 0
}

for file in $(find "$SCRIPT_DIR" -name '*.c' | sort); do
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"

    if [ -n "$FILTER" ]; then
        case "$name" in
            *"$FILTER"*) ;;
            *) continue ;;
        esac
    fi

    # Only files that actually run are useful here: a CHECK-only test has no
    # observable behavior to compare.
    if ! grep -q '// EXIT:\|// OUTPUT:' "$file"; then
        continue
    fi

    # Multi-file tests need their companion sources on the command line.
    extras=""
    file_dir="$(dirname "$file")"
    for ef in $(sed -n 's|.*// EXTRA_FILE: *||p' "$file"); do
        extras="$extras $file_dir/$ef"
    done

    total=$((total + 1))

    if ! build_and_run "$file" "-O0" "$extras"; then
        skipped=$((skipped + 1))
        printf "\nSKIP: %s (does not compile at -O0)\n" "$name"
        continue
    fi
    if ! build_and_run "$file" "-O1" "$extras"; then
        failed=$((failed + 1))
        printf "\nFAIL: %s (compiles at -O0 but not at -O1)\n" "$name"
        head -5 "$WORK/cc.-O1"
        continue
    fi

    status0="$(cat "$WORK/status.-O0")"
    status1="$(cat "$WORK/status.-O1")"

    if [ "$status0" = "124" ] || [ "$status1" = "124" ]; then
        skipped=$((skipped + 1))
        printf "\nSKIP: %s (timeout)\n" "$name"
        continue
    fi

    if [ "$status0" != "$status1" ]; then
        failed=$((failed + 1))
        printf "\nFAIL: %s (exit %s at -O0, %s at -O1)\n" "$name" "$status0" "$status1"
        continue
    fi

    if ! diff -u "$WORK/out.-O0" "$WORK/out.-O1" > "$WORK/diff" 2>&1; then
        failed=$((failed + 1))
        printf "\nFAIL: %s (stdout differs between -O0 and -O1)\n" "$name"
        head -20 "$WORK/diff"
        continue
    fi

    passed=$((passed + 1))

    # ── Reference-compiler leg ────────────────────────────────────────────────
    #
    # Compared against -O1 only: the two blitz levels already agree here, so a
    # disagreement with the reference is a bug in both.
    if [ "$use_oracle" -eq 1 ]; then
        if ! build_and_run_cc "$file" "$extras"; then
            oracle_skipped=$((oracle_skipped + 1))
        else
            oracle_checked=$((oracle_checked + 1))
            status_cc="$(cat "$WORK/status.cc")"
            if [ "$status_cc" = "124" ]; then
                oracle_skipped=$((oracle_skipped + 1))
                oracle_checked=$((oracle_checked - 1))
            elif [ "$status_cc" != "$status1" ]; then
                oracle_failed=$((oracle_failed + 1))
                printf "\nORACLE: %s (exit %s from blitz, %s from %s)\n" \
                    "$name" "$status1" "$status_cc" "$CC"
            elif ! diff -u "$WORK/out.cc" "$WORK/out.-O1" > "$WORK/odiff" 2>&1; then
                oracle_failed=$((oracle_failed + 1))
                printf "\nORACLE: %s (stdout differs from %s)\n" "$name" "$CC"
                head -20 "$WORK/odiff"
            fi
        fi
    fi

    printf "."
done

printf "\n\n%d comparisons: %d matched, %d differed, %d skipped\n" \
    "$total" "$passed" "$failed" "$skipped"
if [ "$use_oracle" -eq 1 ]; then
    printf "%s oracle: %d checked, %d differed, %d skipped\n" \
        "$CC" "$oracle_checked" "$oracle_failed" "$oracle_skipped"
fi
[ "$failed" -eq 0 ] && [ "$oracle_failed" -eq 0 ]
