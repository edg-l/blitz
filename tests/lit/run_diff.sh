#!/bin/sh
# Differential execution harness: compile every runnable lit test at -O0 and
# -O1, run both, and compare exit status and stdout.
#
# This is a self-consistency oracle -- it needs no expected output, only that
# optimization does not change observable behavior. Any mismatch is an
# optimizer miscompile, and the reduced test case is already a .c file that
# belongs in tests/lit/.
#
# The regular suite (run_tests.sh) checks each file against its own directives
# at whatever flags its RUN line names, which for most files is one level only.
# This covers the other one.
#
# Usage:
#   bash tests/lit/run_diff.sh              # every runnable test
#   bash tests/lit/run_diff.sh arithmetic   # only paths matching a substring
#
# Honors BLITZ_VERIFY, so `BLITZ_VERIFY=strict bash tests/lit/run_diff.sh`
# checks IR invariants on both sides of every comparison.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TINYC="${TINYC:-$ROOT/target/debug/tinyc}"
FILTER="${1:-}"

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
    printf "."
done

printf "\n\n%d comparisons: %d matched, %d differed, %d skipped\n" \
    "$total" "$passed" "$failed" "$skipped"
[ "$failed" -eq 0 ]
