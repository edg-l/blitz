#!/bin/sh
# Where does a compile spend its time?
#
#   bash tests/profile.sh <source.c> [tinyc flags...]
#
# Prints the hottest frames by self and by inclusive time, and leaves a
# flamegraph beside the raw samples. Everything lands in
# `$CACHE_DIR/blitz-perf/<name>/` (default `~/.cache`), whose path it prints.
#
# Three things this does that a bare `perf record` does not, each of which cost a
# session to learn:
#
#   Frame pointers. A release-profile build omits them, and `--call-graph fp` then
#   yields stacks that stop at the first frame. `--call-graph dwarf` works but
#   writes ~8KB per sample. So this builds `-C force-frame-pointers=yes` into a
#   target directory of its own, leaving `target/checked` alone -- rebuilding that
#   would also invalidate the symbolization of any earlier recording.
#
#   `perf script`, never `perf report`. The interactive report hangs on these
#   profiles. Symbols are resolved into a text file immediately, while the binary
#   on disk still matches the samples, so a later rebuild cannot spoil them.
#
#   Repetition for short programs. A 200ms compile yields too few samples to read.
#   `REPEAT=n` runs the compile n times under one recording; the aggregate is what
#   matters, not any single run.
#
# Env: REPEAT (default 1), FREQ (samples/sec, default 499), CACHE_DIR.
#
# Needs `perf`. Uses `inferno-flamegraph` and `rustfilt` when installed
# (`cargo install inferno rustfilt`) -- without them the numbers still print, with
# mangled names and no picture.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$1"
if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
    echo "usage: bash tests/profile.sh <source.c> [tinyc flags...]" >&2
    exit 2
fi
shift

REPEAT="${REPEAT:-1}"
FREQ="${FREQ:-499}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache}"
NAME="$(basename "$SRC" .c)$(printf '%s' "$*" | tr -c 'A-Za-z0-9' '_')"
OUT="$CACHE_DIR/blitz-perf/$NAME"
TARGET="$CACHE_DIR/blitz-perf-target"

if ! command -v perf > /dev/null 2>&1; then
    echo "error: perf not found" >&2
    exit 1
fi

mkdir -p "$OUT"

echo "building a frame-pointer tinyc in $TARGET (target/checked is left alone)"
( cd "$ROOT" && CARGO_TARGET_DIR="$TARGET" TMPDIR="$CACHE_DIR/tmp" \
    RUSTFLAGS="-C force-frame-pointers=yes" \
    cargo build --profile checked -p tinyc 2>&1 | tail -1 )
TINYC="$TARGET/checked/tinyc"

echo "recording $REPEAT compile(s) of $SRC $* at ${FREQ}Hz"
perf record -q --call-graph fp -F "$FREQ" -o "$OUT/perf.data" -- \
    sh -c 'i=0; while [ "$i" -lt "$2" ]; do "$1" $4 "$3" -o /dev/null > /dev/null 2>&1 || true; i=$((i + 1)); done' \
    _ "$TINYC" "$REPEAT" "$SRC" "$*" 2>/dev/null || true

# Resolve symbols now: a later rebuild of the profiled binary would leave the
# samples pointing at addresses that no longer mean anything.
perf script -i "$OUT/perf.data" > "$OUT/samples.txt" 2>/dev/null

if command -v inferno-collapse-perf > /dev/null 2>&1; then
    inferno-collapse-perf "$OUT/samples.txt" > "$OUT/folded.txt" 2>/dev/null
    if command -v inferno-flamegraph > /dev/null 2>&1; then
        inferno-flamegraph --title "tinyc $(basename "$SRC") $*" \
            "$OUT/folded.txt" > "$OUT/flame.svg" 2>/dev/null
    fi
else
    # Collapse to `frame;frame;frame count` with awk: one stack per blank-line
    # separated record in `perf script` output, innermost frame first.
    awk '
        /^[^ \t]/ { next }
        /^[ \t]*$/ {
            if (n > 0) { s = ""; for (i = n; i >= 1; i--) s = s (s == "" ? "" : ";") f[i]; print s " 1" }
            n = 0; next
        }
        { sym = $2; for (i = 3; i <= NF; i++) { if ($i ~ /^\(/) break; sym = sym " " $i }
          n++; f[n] = sym }
        END { if (n > 0) { s = ""; for (i = n; i >= 1; i--) s = s (s == "" ? "" : ";") f[i]; print s " 1" } }
    ' "$OUT/samples.txt" > "$OUT/folded.txt"
fi

python3 - "$OUT/folded.txt" > "$OUT/hot.txt" <<'PY'
import collections, sys

self_t = collections.Counter()
incl = collections.Counter()
total = 0
for line in open(sys.argv[1]):
    stack, _, weight = line.rstrip("\n").rpartition(" ")
    try:
        w = int(weight)
    except ValueError:
        continue
    total += w
    frames = stack.split(";")
    self_t[frames[-1]] += w
    for f in set(frames):
        incl[f] += w

if total == 0:
    print("no samples: is the program fast enough to need REPEAT=n?")
    raise SystemExit

def show(title, counter, only_blitz=False, limit=15):
    print(f"-- {title}")
    shown = 0
    for f, w in counter.most_common(2000):
        if only_blitz and "blitz" not in f and "tinyc" not in f:
            continue
        print(f"  {100 * w / total:5.1f}%  {f}")
        shown += 1
        if shown == limit:
            break
    print()

show("self time: where the instructions actually retire", self_t)
show("inclusive: which pass owns that work", incl, only_blitz=True)
PY

if command -v rustfilt > /dev/null 2>&1; then
    rustfilt < "$OUT/hot.txt" > "$OUT/hot.demangled.txt"
    mv "$OUT/hot.demangled.txt" "$OUT/hot.txt"
fi

cut -c1-140 "$OUT/hot.txt"
echo "samples, folded stacks, flamegraph and this list: $OUT"
