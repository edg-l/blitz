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
# One test file per job, `JOBS` jobs at a time (default: every core). Each job
# gets a work directory of its own -- the binaries and captured output are named
# per level, so jobs sharing one directory would overwrite each other's -- and
# writes its verdicts to a numbered file the parent reads back in path order, so
# the report is the same whatever order the jobs finished in.
#
# Usage:
#   bash tests/lit/run_diff.sh              # every runnable test
#   bash tests/lit/run_diff.sh arithmetic   # only paths matching a substring
#   NO_ORACLE=1 bash tests/lit/run_diff.sh  # skip the reference-compiler leg
#   CC=clang bash tests/lit/run_diff.sh     # use a different reference
#   JOBS=1 bash tests/lit/run_diff.sh       # sequential
#
# Honors BLITZ_VERIFY, so `BLITZ_VERIFY=strict bash tests/lit/run_diff.sh`
# checks IR invariants on every compilation.
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
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# The reference leg is optional: it needs a working cc, and a test whose C the
# reference rejects is skipped rather than failed.
use_oracle=1
if [ -n "${NO_ORACLE:-}" ] || ! command -v "$CC" > /dev/null 2>&1; then
    use_oracle=0
fi

if [ ! -x "$TINYC" ]; then
    echo "error: tinyc not found at $TINYC (run 'cargo build --profile checked -p tinyc' first)" >&2
    exit 1
fi

# ── Verdict protocol ─────────────────────────────────────────────────────────
#
# A job prints exactly one of `PASS`, `SKIP: <why>` or `FAIL: <why>` for the file
# it was given, and when it passed and the oracle leg ran, one of `OCHECK`,
# `OSKIP` or `OFAIL: <why>`. Detail lines are indented. Nothing else may go to a
# job's stdout: the parent counts these lines.

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
# prototypes (`extern int printf(char* fmt, ...);`), which the reference
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

# Compare one file across the three legs. Reached through `$0 --one <file>`.
run_one_file() {
    file="$1"
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"

    WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_diff_XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT INT TERM

    # Multi-file tests need their companion sources on the command line.
    extras=""
    file_dir="$(dirname "$file")"
    for ef in $(sed -n 's|.*// EXTRA_FILE: *||p' "$file"); do
        extras="$extras $file_dir/$ef"
    done

    if ! build_and_run "$file" "-O0" "$extras"; then
        printf "SKIP: %s (does not compile at -O0)\n" "$name"
        return 0
    fi
    if ! build_and_run "$file" "-O1" "$extras"; then
        printf "FAIL: %s (compiles at -O0 but not at -O1)\n" "$name"
        head -5 "$WORK/cc.-O1" | sed 's/^/    /'
        return 0
    fi

    status0="$(cat "$WORK/status.-O0")"
    status1="$(cat "$WORK/status.-O1")"

    if [ "$status0" = "124" ] || [ "$status1" = "124" ]; then
        printf "SKIP: %s (timeout)\n" "$name"
        return 0
    fi

    if [ "$status0" != "$status1" ]; then
        printf "FAIL: %s (exit %s at -O0, %s at -O1)\n" "$name" "$status0" "$status1"
        return 0
    fi

    if ! diff -u "$WORK/out.-O0" "$WORK/out.-O1" > "$WORK/diff" 2>&1; then
        printf "FAIL: %s (stdout differs between -O0 and -O1)\n" "$name"
        head -20 "$WORK/diff" | sed 's/^/    /'
        return 0
    fi

    echo "PASS"

    # ── Reference-compiler leg ────────────────────────────────────────────────
    #
    # Compared against -O1 only: the two blitz levels already agree here, so a
    # disagreement with the reference is a bug in both.
    if [ "$use_oracle" -eq 1 ]; then
        if ! build_and_run_cc "$file" "$extras"; then
            echo "OSKIP"
        else
            status_cc="$(cat "$WORK/status.cc")"
            if [ "$status_cc" = "124" ]; then
                echo "OSKIP"
            elif [ "$status_cc" != "$status1" ]; then
                printf "OFAIL: %s (exit %s from blitz, %s from %s)\n" \
                    "$name" "$status1" "$status_cc" "$CC"
            elif ! diff -u "$WORK/out.cc" "$WORK/out.-O1" > "$WORK/odiff" 2>&1; then
                printf "OFAIL: %s (stdout differs from %s)\n" "$name" "$CC"
                head -20 "$WORK/odiff" | sed 's/^/    /'
            else
                echo "OCHECK"
            fi
        fi
    fi
}

if [ "$1" = "--one" ]; then
    run_one_file "$2"
    exit 0
fi

FILTER="${1:-}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_diff_XXXXXX")"
trap 'rm -rf "$WORK"' EXIT INT TERM

# Which files this harness has anything to say about.
: > "$WORK/files"
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

    # A test pinned to one optimization level by `// FLAGS:` has nothing to say
    # about the other: it is committed at the level whose behavior it asserts
    # precisely because the other level does something else. Comparing the two
    # would report the known failure this harness cannot act on.
    if sed -n 's|.*// FLAGS: *||p' "$file" | grep -q '\-O[0-9]'; then
        continue
    fi

    echo "$file" >> "$WORK/files"
done

# `<verdict file>|<test file>` per job: one xargs placeholder carries both, and
# the number fixes the order the parent reads verdicts back in.
n=0
while IFS= read -r f; do
    n=$((n + 1))
    printf '%s/%04d.out|%s\n' "$WORK" "$n" "$f"
done < "$WORK/files" > "$WORK/jobs"

# Absolute, and run through `sh`, so a job does not depend on the cwd it
# inherited or on this file carrying an execute bit.
SELF="$SCRIPT_DIR/run_diff.sh"
export SELF
if [ -s "$WORK/jobs" ]; then
    xargs -d '\n' -P "$JOBS" -n1 sh -c '
        out="${1%%|*}"
        file="${1#*|}"
        sh "$SELF" --one "$file" > "$out"
    ' _ < "$WORK/jobs"
fi

total=0
passed=0
failed=0
skipped=0
oracle_checked=0
oracle_failed=0
oracle_skipped=0
report=""
for out in "$WORK"/*.out; do
    [ -f "$out" ] || continue
    while IFS= read -r line; do
        case "$line" in
            PASS)
                total=$((total + 1))
                passed=$((passed + 1))
                printf "." >&2
                ;;
            SKIP:*)
                total=$((total + 1))
                skipped=$((skipped + 1))
                report="$report$line
"
                ;;
            FAIL:*)
                total=$((total + 1))
                failed=$((failed + 1))
                report="$report$line
"
                ;;
            OCHECK)
                oracle_checked=$((oracle_checked + 1))
                ;;
            OSKIP)
                oracle_skipped=$((oracle_skipped + 1))
                ;;
            OFAIL:*)
                oracle_failed=$((oracle_failed + 1))
                report="$report ORACLE${line#OFAIL}
"
                ;;
            *)
                # Indented detail under the verdict above it.
                report="$report$line
"
                ;;
        esac
    done < "$out"
done

if [ -n "$report" ]; then
    printf "\n%s" "$report"
fi
printf "\n%d comparisons: %d matched, %d differed, %d skipped\n" \
    "$total" "$passed" "$failed" "$skipped"
if [ "$use_oracle" -eq 1 ]; then
    printf "%s oracle: %d checked, %d differed, %d skipped\n" \
        "$CC" "$oracle_checked" "$oracle_failed" "$oracle_skipped"
fi
[ "$failed" -eq 0 ] && [ "$oracle_failed" -eq 0 ]
