#!/bin/sh
# Does a change emit exactly the same code as the compiler it started from?
#
# For every (program, optimization level) pair it disassembles with both
# compilers and compares the text. The answer is one of four, and each says
# something different:
#
#   identical    same disassembly
#   asm          both compiled, different code   <- a behaviour change
#   status       one compiled and the other did not
#   both-failed  neither compiled, and with the same message
#
# This is the check for a change that is meant to alter no output at all -- a
# refactor, a data-structure swap, a pass made faster. It is stronger than a pass
# count, which cannot tell "unchanged" from "changed in a way no test covers", and
# it is the wrong tool for a change that is meant to emit better code; that one
# belongs to tests/fuzz/compare_ref.sh, which judges behaviour per (seed, level).
#
# Usage:
#   bash tests/run_identity.sh <ref-tinyc> <new-tinyc> [file...]
#
# With no files it takes the lit corpus. To include generated programs, pass them:
#   python3 tests/fuzz/gen_c.py --seed 7 --shape pressure --out /tmp/p7.c
#   bash tests/run_identity.sh /tmp/ref/tinyc target/checked/tinyc /tmp/p7.c
#
# Building the reference, which must be a checkout of the ref plus ANY change of
# this tree's that alters how a disassembly is formatted -- otherwise every
# comparison differs on that alone and the run says nothing:
#   git worktree add ~/.cache/blitz-ref <ref>
#   cd ~/.cache/blitz-ref && cargo build --profile checked -p tinyc
#
# `JOBS` jobs at a time (default: every core). One job per (file, level) rather
# than per file, because a single slow program otherwise serializes both of its
# levels and becomes the whole wall-clock.
#
# Each job compiles into a directory of its own: given no -o, tinyc writes
# ./a.out, and two jobs sharing it produce write errors that read as behaviour
# differences.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
LEVELS="${LEVELS:--O0 -O1}"

# A job is this script again, with the two compilers arriving through the
# environment rather than the argument list.
if [ "$1" = "--one" ]; then
    mode=worker
    one_file="$2"
    one_level="$3"
else
    mode=parent
    REF="$1"
    NEW="$2"
    if [ -z "$REF" ] || [ -z "$NEW" ]; then
        echo "usage: bash tests/run_identity.sh <ref-tinyc> <new-tinyc> [file...]" >&2
        exit 2
    fi
    shift 2
    for bin in "$REF" "$NEW"; do
        if [ ! -x "$bin" ]; then
            echo "error: $bin is not executable" >&2
            exit 1
        fi
    done
fi

# ── One comparison ───────────────────────────────────────────────────────────
#
# Prints one verdict line. Reached through `$0 --one <file> <level>`.
compare_one() {
    file="$1"
    level="$2"

    WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_ident_XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT INT TERM

    ref_out="$("$REF" "$file" "$level" --emit-asm -o "$WORK/ref.bin" 2>/dev/null)" && ref_status=0 || ref_status=$?
    new_out="$("$NEW" "$file" "$level" --emit-asm -o "$WORK/new.bin" 2>/dev/null)" && new_status=0 || new_status=$?

    if [ "$ref_status" -ne "$new_status" ]; then
        printf "STATUS %s %s (ref=%s new=%s)\n" "$file" "$level" "$ref_status" "$new_status"
    elif [ "$ref_status" -ne 0 ]; then
        printf "BOTHFAIL %s %s\n" "$file" "$level"
    elif [ "$ref_out" = "$new_out" ]; then
        printf "SAME %s %s\n" "$file" "$level"
    else
        printf "ASM %s %s\n" "$file" "$level"
    fi
}

if [ "$mode" = worker ]; then
    compare_one "$one_file" "$one_level"
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_ident_XXXXXX")"
trap 'rm -rf "$WORK"' EXIT INT TERM

if [ $# -gt 0 ]; then
    printf '%s\n' "$@" > "$WORK/files"
else
    find "$SCRIPT_DIR/lit" -name '*.c' | sort > "$WORK/files"
fi

# `<file>|<level>` per job. `|` cannot appear in a path or a level.
: > "$WORK/jobs"
while IFS= read -r f; do
    for level in $LEVELS; do
        printf '%s|%s\n' "$f" "$level" >> "$WORK/jobs"
    done
done < "$WORK/files"

# Absolute, and run through `sh`, so a job does not depend on the cwd it
# inherited or on this file carrying an execute bit.
SELF="$SCRIPT_DIR/run_identity.sh"
export SELF REF NEW
xargs -d '\n' -P "$JOBS" -n1 sh -c '
    file="${1%%|*}"
    level="${1#*|}"
    sh "$SELF" --one "$file" "$level"
' _ < "$WORK/jobs" > "$WORK/verdicts"

same=$(grep -c '^SAME ' "$WORK/verdicts" || true)
bothfail=$(grep -c '^BOTHFAIL ' "$WORK/verdicts" || true)
asm=$(grep -c '^ASM ' "$WORK/verdicts" || true)
status=$(grep -c '^STATUS ' "$WORK/verdicts" || true)

if [ "$asm" -gt 0 ] || [ "$status" -gt 0 ]; then
    printf "\n"
    grep -E '^(ASM|STATUS) ' "$WORK/verdicts" | sort
fi
printf "\n%d comparisons: %d identical, %d differ, %d differ in status, %d failed on both\n" \
    "$((same + bothfail + asm + status))" "$same" "$asm" "$status" "$bothfail"
[ "$asm" -eq 0 ] && [ "$status" -eq 0 ]
