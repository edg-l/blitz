#!/bin/sh
# Run all lit tests. Requires tinyc and blitztest on PATH or in target/debug/.
#
# One test file per job, `JOBS` jobs at a time (default: every core). A file's own
# CHECK/EXIT/OUTPUT tests still run in order inside its job, since they share the
# compile flags parsed from it.
#
# Each job writes its verdicts to a file of its own and the parent reads them back
# in path order, so failures are reported in the same order whatever the jobs did,
# and a multi-line diff cannot interleave with another job's. `JOBS=1` is the
# sequential run, for when a failure might be the parallelism itself.
#
# A test that shells out to `objdump` needs `test_utils::objdump_disasm`'s temp
# path to be unique per process, or concurrent jobs disassemble each other's code.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The `checked` profile: optimized with `debug_assertions` on. A debug blitz is
# ~10x slower and the assertions are what catch a broken internal invariant, so
# neither plain profile is the right one to test against. Build it with
# `cargo build --profile checked -p tinyc -p blitztest`.
PROFILE="${PROFILE:-checked}"
TINYC="${TINYC:-$ROOT/target/$PROFILE/tinyc}"
BLITZTEST="${BLITZTEST:-$ROOT/target/$PROFILE/blitztest}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

if [ ! -x "$TINYC" ]; then
    echo "error: tinyc not found at $TINYC (run 'cargo build --profile checked -p tinyc' first)" >&2
    exit 1
fi
if [ ! -x "$BLITZTEST" ]; then
    echo "error: blitztest not found at $BLITZTEST (run 'cargo build --profile checked -p blitztest' first)" >&2
    exit 1
fi

# ── Verdict protocol ─────────────────────────────────────────────────────────
#
# A job prints one `PASS` or `FAIL: <name> (<why>)` line per test, plus any
# indented detail lines under a FAIL. The parent counts the lines and prints the
# FAILs. Nothing else may go to a job's stdout.

emit_pass() {
    echo "PASS"
    printf "." >&2
}

emit_fail() {
    printf "FAIL: %s\n" "$1"
    printf "F" >&2
}

run_check_test() {
    file="$1"
    mode="$2"
    extra_flags="$3"
    passes="$4"
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"

    if BLITZ_PASSES="$passes" "$TINYC" "$file" $extra_flags "$mode" 2>&1 | "$BLITZTEST" "$file" 2>/dev/null; then
        emit_pass
    else
        emit_fail "$name"
    fi
}

run_exit_test() {
    file="$1"
    expected="$2"
    passes="$3"
    shift 3
    extras="$*"
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"

    tmpfile="$(mktemp /tmp/blitztest_XXXXXX)"

    if BLITZ_PASSES="$passes" "$TINYC" "$file" $extras -o "$tmpfile" 2>/dev/null; then
        actual=0
        timeout 10 "$tmpfile" 2>/dev/null && actual=0 || actual=$?
        rm -f "$tmpfile"
        if [ "$actual" -eq 124 ]; then
            emit_fail "$name (timeout)"
            return
        fi
        if [ "$actual" -eq "$expected" ]; then
            emit_pass
        else
            emit_fail "$name (expected exit $expected, got $actual)"
        fi
    else
        rm -f "$tmpfile"
        emit_fail "$name (compilation failed)"
    fi
}

run_output_test() {
    file="$1"
    passes="$2"
    shift 2
    extras="$*"
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"

    tmpfile="$(mktemp /tmp/blitztest_XXXXXX)"
    outfile="$(mktemp /tmp/blitztest_out_XXXXXX)"
    expectfile="$(mktemp /tmp/blitztest_exp_XXXXXX)"

    # Extract expected output lines from // OUTPUT: directives
    sed -n 's|.*// OUTPUT: \(.*\)|\1|p' "$file" > "$expectfile"

    if BLITZ_PASSES="$passes" "$TINYC" "$file" $extras -o "$tmpfile" 2>/dev/null; then
        timeout 10 "$tmpfile" > "$outfile" 2>/dev/null
        actual=$?
        if [ "$actual" -eq 124 ]; then
            rm -f "$tmpfile" "$outfile" "$expectfile"
            emit_fail "$name (timeout)"
            return
        fi
        if diff -u "$expectfile" "$outfile" > /dev/null 2>&1; then
            emit_pass
        else
            emit_fail "$name (output mismatch)"
            diff -u "$expectfile" "$outfile" | head -20 | sed 's/^/    /'
        fi
        rm -f "$tmpfile" "$outfile" "$expectfile"
    else
        rm -f "$tmpfile" "$outfile" "$expectfile"
        emit_fail "$name (compilation failed)"
    fi
}

# Run every test one file declares. Reached through `$0 --one <file>`, which is
# how a job is started.
run_one_file() {
    file="$1"
    has_check=false
    has_exit=false
    has_output=false
    exit_code=0
    mode=""

    extra_files=""
    check_flags=""
    # Deviating from an opt level is an environment setting, not a compiler flag:
    # `-O0`/`-O1` are the configurations this compiler claims to compile
    # correctly, and BLITZ_PASSES is a debugging facility beside BLITZ_DEBUG.
    # A test that needs one writes `// PASSES: -inlining`.
    test_passes=""
    run_flags=""
    file_dir="$(dirname "$file")"

    while IFS= read -r line; do
        case "$line" in
            *"// CHECK:"*|*"// CHECK-"*)
                has_check=true
                ;;
            *"// EXIT:"*)
                has_exit=true
                exit_code="$(echo "$line" | sed 's/.*\/\/ EXIT: *//')"
                ;;
            *"// OUTPUT:"*)
                has_output=true
                ;;
            *"// RUN:"*"--emit-ir"*)
                mode="--emit-ir"
                # Extract the opt level from the RUN line. Deviating from a
                # level is `BLITZ_PASSES=` in the environment, picked up below.
                check_flags="$(echo "$line" | sed 's|.*// RUN:||' | grep -oE '\-O[0-9]' | tr '\n' ' ')"
                ;;
            *"// RUN:"*"--emit-asm"*)
                mode="--emit-asm"
                check_flags="$(echo "$line" | sed 's|.*// RUN:||' | grep -oE '\-O[0-9]' | tr '\n' ' ')"
                ;;
            *"// PASSES:"*)
                test_passes="$(echo "$line" | sed 's/.*\/\/ PASSES: *//')"
                ;;
            *"// EXTRA_FILE:"*)
                ef="$(echo "$line" | sed 's/.*\/\/ EXTRA_FILE: *//')"
                extra_files="$extra_files $file_dir/$ef"
                ;;
            *"// FLAGS:"*)
                # Compiler flags for the EXIT/OUTPUT runs. `// RUN:` carries them
                # for CHECK tests, but a behavioural test had no way to ask for
                # an opt level, so one that only holds at -O0 could not be
                # committed at all.
                run_flags="$(echo "$line" | sed 's/.*\/\/ FLAGS: *//')"
                ;;
        esac
    done < "$file"

    if [ "$has_check" = true ] && [ -n "$mode" ]; then
        run_check_test "$file" "$mode" "$check_flags" "$test_passes"
    fi
    if [ "$has_exit" = true ]; then
        run_exit_test "$file" "$exit_code" "$test_passes" $run_flags $extra_files
    fi
    if [ "$has_output" = true ]; then
        run_output_test "$file" "$test_passes" $run_flags $extra_files
    fi
}

if [ "$1" = "--one" ]; then
    # `set +e` so a failing check inside cannot end the job before its verdict
    # is written: the verdict file is how the parent learns this test ran at
    # all, and a job that dies early takes the test out of the count rather
    # than into the failure list.
    set +e
    run_one_file "$2"
    exit 0
fi

WORK="$(mktemp -d /tmp/blitzlit_XXXXXX)"
trap 'rm -rf "$WORK"' EXIT INT TERM

# `find | sort` fixes the order verdicts are read back in; the jobs themselves
# finish in whatever order they finish.
find "$SCRIPT_DIR" -name '*.c' | sort > "$WORK/files"

# Each job line is `<verdict file>|<test file>`: one xargs placeholder carries
# both, and the number fixes the order the parent reads verdicts back in. `|`
# cannot appear in these paths.
n=0
while IFS= read -r f; do
    n=$((n + 1))
    printf '%s/%04d.out|%s\n' "$WORK" "$n" "$f"
done < "$WORK/files" > "$WORK/jobs"

# Absolute, and run through `sh`, so a job does not depend on the cwd it
# inherited or on this file carrying an execute bit.
#
# `|| true` because a failing job makes xargs exit 123, which under `set -e`
# ends the run before the loop below reads a single verdict: the output was
# then a wall of dots ending in `F` with no summary and no test named. The
# verdicts live in files, so the exit status here carries nothing.
SELF="$SCRIPT_DIR/run_tests.sh"
export SELF
xargs -d '\n' -P "$JOBS" -n1 sh -c '
    out="${1%%|*}"
    file="${1#*|}"
    sh "$SELF" --one "$file" > "$out"
' _ < "$WORK/jobs" || true

passed=0
failed=0
total=0
fails=""
for out in "$WORK"/*.out; do
    [ -f "$out" ] || continue
    while IFS= read -r line; do
        case "$line" in
            PASS)
                passed=$((passed + 1))
                total=$((total + 1))
                ;;
            FAIL:*)
                failed=$((failed + 1))
                total=$((total + 1))
                fails="$fails$line
"
                ;;
            *)
                # Indented detail under the FAIL above it.
                fails="$fails$line
"
                ;;
        esac
    done < "$out"
done

printf "\n"
if [ -n "$fails" ]; then
    printf "\n%s" "$fails"
fi
printf "\n%d tests: %d passed, %d failed\n" "$total" "$passed" "$failed"
[ "$failed" -eq 0 ]
