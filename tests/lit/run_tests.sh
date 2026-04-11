#!/bin/sh
# Run all lit tests. Requires tinyc and blitztest on PATH or in target/debug/.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TINYC="${TINYC:-$ROOT/target/debug/tinyc}"
BLITZTEST="${BLITZTEST:-$ROOT/target/debug/blitztest}"

if [ ! -x "$TINYC" ]; then
    echo "error: tinyc not found at $TINYC (run 'cargo build -p tinyc' first)" >&2
    exit 1
fi
if [ ! -x "$BLITZTEST" ]; then
    echo "error: blitztest not found at $BLITZTEST (run 'cargo build -p blitztest' first)" >&2
    exit 1
fi

passed=0
failed=0
total=0

run_check_test() {
    local file="$1"
    local mode="$2"
    local name
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"
    total=$((total + 1))

    if "$TINYC" "$file" "$mode" 2>&1 | "$BLITZTEST" "$file" 2>/dev/null; then
        passed=$((passed + 1))
        printf "."
    else
        failed=$((failed + 1))
        printf "\nFAIL: %s\n" "$name"
    fi
}

run_exit_test() {
    local file="$1"
    local expected="$2"
    shift 2
    local extras="$*"
    local name
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"
    total=$((total + 1))

    local tmpfile
    tmpfile="$(mktemp /tmp/blitztest_XXXXXX)"

    if "$TINYC" "$file" $extras -o "$tmpfile" 2>/dev/null; then
        local actual=0
        timeout 10 "$tmpfile" 2>/dev/null && actual=0 || actual=$?
        rm -f "$tmpfile"
        if [ "$actual" -eq 124 ]; then
            failed=$((failed + 1))
            printf "\nFAIL: %s (timeout)\n" "$name"
            return
        fi
        if [ "$actual" -eq "$expected" ]; then
            passed=$((passed + 1))
            printf "."
        else
            failed=$((failed + 1))
            printf "\nFAIL: %s (expected exit %d, got %d)\n" "$name" "$expected" "$actual"
        fi
    else
        rm -f "$tmpfile"
        failed=$((failed + 1))
        printf "\nFAIL: %s (compilation failed)\n" "$name"
    fi
}

run_output_test() {
    local file="$1"
    shift
    local extras="$*"
    local name
    name="$(echo "$file" | sed "s|^$SCRIPT_DIR/||")"
    total=$((total + 1))

    local tmpfile
    tmpfile="$(mktemp /tmp/blitztest_XXXXXX)"
    local outfile
    outfile="$(mktemp /tmp/blitztest_out_XXXXXX)"
    local expectfile
    expectfile="$(mktemp /tmp/blitztest_exp_XXXXXX)"

    # Extract expected output lines from // OUTPUT: directives
    sed -n 's|.*// OUTPUT: \(.*\)|\1|p' "$file" > "$expectfile"

    if "$TINYC" "$file" $extras -o "$tmpfile" 2>/dev/null; then
        timeout 10 "$tmpfile" > "$outfile" 2>/dev/null
        local actual=$?
        if [ "$actual" -eq 124 ]; then
            rm -f "$tmpfile" "$outfile" "$expectfile"
            failed=$((failed + 1))
            printf "\nFAIL: %s (timeout)\n" "$name"
            return
        fi
        if diff -u "$expectfile" "$outfile" > /dev/null 2>&1; then
            passed=$((passed + 1))
            printf "."
        else
            failed=$((failed + 1))
            printf "\nFAIL: %s (output mismatch)\n" "$name"
            diff -u "$expectfile" "$outfile" | head -20
        fi
        rm -f "$tmpfile" "$outfile" "$expectfile"
    else
        rm -f "$tmpfile" "$outfile" "$expectfile"
        failed=$((failed + 1))
        printf "\nFAIL: %s (compilation failed)\n" "$name"
    fi
}

# Find and run all .c test files
for file in $(find "$SCRIPT_DIR" -name '*.c' | sort); do
    # Parse directives from the file
    has_check=false
    has_exit=false
    has_output=false
    exit_code=0
    mode=""

    extra_files=""
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
                ;;
            *"// RUN:"*"--emit-asm"*)
                mode="--emit-asm"
                ;;
            *"// EXTRA_FILE:"*)
                ef="$(echo "$line" | sed 's/.*\/\/ EXTRA_FILE: *//')"
                extra_files="$extra_files $file_dir/$ef"
                ;;
        esac
    done < "$file"

    if [ "$has_check" = true ] && [ -n "$mode" ]; then
        run_check_test "$file" "$mode"
    fi
    if [ "$has_exit" = true ]; then
        run_exit_test "$file" "$exit_code" $extra_files
    fi
    if [ "$has_output" = true ]; then
        run_output_test "$file" $extra_files
    fi
done

printf "\n\n%d tests: %d passed, %d failed\n" "$total" "$passed" "$failed"
[ "$failed" -eq 0 ]
