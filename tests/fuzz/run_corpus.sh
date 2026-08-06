#!/bin/sh
# Check blitz against the saved corpus in tests/fuzz/corpus/.
#
# Same three oracles as run_fuzz.sh (oracles.sh holds them), over programs that
# are kept rather than generated. The point is width per second: a 200-seed
# sweep is minutes and so is not run between every change, and at the 30 seeds
# that every gate does run, all three shapes were green while seven programs
# miscompiled. **A session can work all day, see every gate pass, and never
# learn that.** A program that has ever been wrong belongs here, where checking
# it again costs a second.
#
# Usage:
#   bash tests/fuzz/run_corpus.sh [dir]
#
# Each program carries its own answer in a `// OUTPUT:` directive written by
# gen_c.py when it was generated, so the corpus needs no separate expectations
# file and does not depend on gen_c.py still producing that program.
#
# A program is saved with its verdict in the filename's directory:
#
#   corpus/fixed/     was wrong, now right -- a regression target
#   corpus/open/      still wrong, and expected to be
#
# Both are run. `open` failures are reported and counted but do not fail the
# run, because a known-open bug is not news; `fixed` failures do. An `open`
# program that PASSES is reported too -- it means a bug was fixed without the
# corpus being updated, and the file should move to `fixed/`.
#
# Honors BLITZ_VERIFY, CC, COMPILE_TIMEOUT, PROFILE, TINYC and RESULTS exactly
# as run_fuzz.sh does.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROFILE="${PROFILE:-checked}"
TINYC="${TINYC:-$ROOT/target/$PROFILE/tinyc}"
CC="${CC:-cc}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-60}"
CORPUS="${1:-$SCRIPT_DIR/corpus}"

if [ ! -x "$TINYC" ]; then
    echo "error: tinyc not found at $TINYC (run 'cargo build --profile checked -p tinyc' first)" >&2
    exit 1
fi
if [ ! -d "$CORPUS" ]; then
    echo "error: no corpus directory at $CORPUS" >&2
    exit 1
fi

. "$SCRIPT_DIR/oracles.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/blitz_corpus_XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [ -n "$RESULTS" ]; then
    : > "$RESULTS"
fi

pass=0
fail=0
skip=0
open_fail=0
open_pass=0

for src in "$CORPUS"/fixed/*.c "$CORPUS"/open/*.c; do
    [ -e "$src" ] || continue
    rel="${src#"$CORPUS"/}"
    stem="$(basename "$src" .c)"
    case "$rel" in
        open/*) expected=fail ;;
        *)      expected=pass ;;
    esac

    st=0
    check_program "$stem" "$rel" "$src" || st=$?
    case "$st" in
        0)
            if [ "$expected" = fail ]; then
                open_pass=$((open_pass + 1))
                printf "\nNEWS %s: an open bug now passes -- move it to corpus/fixed/\n" "$rel"
            else
                pass=$((pass + 1)); printf "."
            fi
            ;;
        2) skip=$((skip + 1)) ;;
        *)
            if [ "$expected" = fail ]; then
                open_fail=$((open_fail + 1))
            else
                fail=$((fail + 1))
            fi
            ;;
    esac
done

printf "\n\ncorpus: %d passed, %d REGRESSED, %d still-open, %d newly-passing, %d skipped\n" \
    "$pass" "$fail" "$open_fail" "$open_pass" "$skip"

# A newly-passing open program is not a failure, but leaving it in `open/` means
# the next regression there reads as expected. Say so loudly and move on.
[ "$fail" -eq 0 ]
