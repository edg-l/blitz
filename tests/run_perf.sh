#!/usr/bin/env bash
# How fast the emitted code actually is, in CPU cycles, against gcc/clang -O2.
#
#   bash tests/run_perf.sh              # the ranking
#   bash tests/run_perf.sh --raw        # per-program counters, no ratios
#
# Environment:
#   PROFILE=checked      which build of tinyc (default checked)
#   TINYC=path           override the compiler under test
#   PERF_LEVEL=-O1       the blitz level to measure (default -O1)
#   PERF_CC="gcc clang"  the reference compilers
#   RUNS=5               samples per program; the median is reported
#   ARGS=100             arguments passed, which is what scales the kernels
#
# THIS IS THE RANKING, and `run_codesize.sh` is not.  Instruction counts --
# static or retired -- invert on any change that trades an instruction for
# latency, which is a trade worth making and one gcc makes routinely: `x * 7`
# as `shl; sub` retires one more instruction than `imul` and costs 1.1% fewer
# cycles.  A metric that scores that as a regression cannot rank a compiler
# whose goal is the best possible machine code, so cycles is what is ranked and
# instruction counts are reported beside it as diagnostics.
#
# IPC IS NOT A GOAL EITHER, for the same reason from the other side: a compiler
# emitting more, cheaper instructions raises IPC while doing more work.  It is
# printed because it says *why* a kernel is slow -- high IPC with high cycles is
# too much work, low IPC with high cycles is stalls -- not because more is
# better.
#
# Only the `live` corpus is measured.  Every other corpus computes an answer
# from no runtime input, so `gcc -O2` folds the program to a constant and there
# is nothing left to time.  `live` seeds each kernel from `argc`, and the same
# `argc` is what sets the repeat count, so passing arguments scales the work
# without letting any compiler evaluate it ahead of time.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE="${PROFILE:-checked}"
TINYC="${TINYC:-$ROOT/target/$PROFILE/tinyc}"
PERF_LEVEL="${PERF_LEVEL:--O1}"
PERF_CC="${PERF_CC:-gcc clang}"
RUNS="${RUNS:-5}"
ARGS="${ARGS:-100}"

raw=0
[ "${1:-}" = "--raw" ] && raw=1

if [ ! -x "$TINYC" ]; then
    echo "no tinyc at $TINYC" >&2
    echo "build it with: cargo build --profile $PROFILE -p tinyc" >&2
    exit 2
fi
if ! command -v perf >/dev/null; then
    echo "perf not found; this harness measures hardware counters" >&2
    exit 2
fi
if ! perf stat -e cycles true >/dev/null 2>&1; then
    echo "perf cannot read counters here (kernel.perf_event_paranoid is $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null))" >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
# The kernels take their repeat count from argc, so the arguments are the dial.
set -- $(seq "$ARGS")
RUN_ARGS="$*"

# Median cycles and instructions for one executable, as "cycles instructions".
measure() {
    exe="$1"
    : > "$WORK/samples"
    i=0
    while [ "$i" -lt "$RUNS" ]; do
        # shellcheck disable=SC2086 -- RUN_ARGS is a list of arguments on purpose.
        perf stat -x, -e cycles,instructions "$exe" $RUN_ARGS 2>&1 >/dev/null \
            | awk -F, '$3=="cycles"{c=$1} $3=="instructions"{n=$1} END{if (c && n) print c, n}' \
            >> "$WORK/samples"
        i=$((i + 1))
    done
    awk '{c[NR]=$1; n[NR]=$2} END {
        if (NR == 0) { print "- -"; exit }
        asort_c = 0
        for (i = 1; i <= NR; i++) for (j = i+1; j <= NR; j++) {
            if (c[j] < c[i]) { t=c[i]; c[i]=c[j]; c[j]=t }
            if (n[j] < n[i]) { t=n[i]; n[i]=n[j]; n[j]=t }
        }
        m = int((NR+1)/2); print c[m], n[m]
    }' "$WORK/samples"
}

printf '%-18s %14s %7s' "program" "cycles" "IPC"
for cc in $PERF_CC; do printf ' %10s' "vs $cc"; done
printf '\n'

: > "$WORK/ratios"
skipped=0
for file in "$ROOT"/tests/lit/live/*.c; do
    name="$(basename "$file")"
    if ! "$TINYC" "$PERF_LEVEL" "$file" -o "$WORK/blitz" >/dev/null 2>&1; then
        printf '%-18s %14s   (blitz failed to compile it)\n' "$name" "-"
        skipped=$((skipped + 1))
        continue
    fi
    # shellcheck disable=SC2086
    want="$("$WORK/blitz" $RUN_ARGS)"
    read -r bc bi <<EOF
$(measure "$WORK/blitz")
EOF
    printf '%-18s %14s %7.2f' "$name" "$bc" "$(awk -v a="$bi" -v b="$bc" 'BEGIN{print a/b}')"

    for cc in $PERF_CC; do
        if ! "$cc" -O2 -w "$file" -o "$WORK/$cc" >/dev/null 2>&1; then
            printf ' %10s' "-"
            continue
        fi
        # A wrong answer must not score. Cycles say nothing about a program
        # that computed something else.
        # shellcheck disable=SC2086
        if [ "$("$WORK/$cc" $RUN_ARGS)" != "$want" ]; then
            printf ' %10s' "DIFFERS"
            echo "$name $cc OUTPUT-DIFFERS" >> "$WORK/bad"
            continue
        fi
        read -r rc _ri <<EOF
$(measure "$WORK/$cc")
EOF
        printf ' %10s' "$(awk -v a="$bc" -v b="$rc" 'BEGIN{printf "x%.2f", a/b}')"
        echo "$cc $(awk -v a="$bc" -v b="$rc" 'BEGIN{print a/b}')" >> "$WORK/ratios"
    done
    printf '\n'
done

[ "$raw" = 1 ] && exit 0

echo
for cc in $PERF_CC; do
    awk -v cc="$cc" '$1==cc {s += log($2); n++} END {
        if (n) printf "blitz %s vs %s -O2: x%.3f cycles geomean over %d programs\n", ENVIRON["PERF_LEVEL"], cc, exp(s/n), n
    }' "$WORK/ratios"
done
[ "$skipped" -gt 0 ] && echo "($skipped program(s) blitz could not compile, excluded)"
[ -f "$WORK/bad" ] && { echo "OUTPUT MISMATCHES -- these are correctness bugs, not slow code:"; cat "$WORK/bad"; }
exit 0
