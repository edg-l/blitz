#!/bin/sh
# The three-oracle check of one C program, shared by every driver that runs one.
#
#   predicted   a `// OUTPUT:` directive, which gen_c.py writes by interpreting
#               the program as it builds it -- an answer that predates any
#               compiler
#   -O0 vs -O1  self-consistency: optimization must not change behavior
#   vs cc       ground truth, catching bugs that are equally wrong at both
#               optimization levels
#
# Sourced, not executed. `run_fuzz.sh` supplies freshly generated programs and
# `run_corpus.sh` supplies saved ones; the oracles must not differ between them,
# or a program promoted from one to the other changes verdict on the move.
#
# Callers set: TINYC, CC, COMPILE_TIMEOUT, WORK, and optionally RESULTS.

# One line per program and level when RESULTS is set, so a run can be compared
# with another instead of read.
record() {
    if [ -n "$RESULTS" ]; then
        printf "%s %s %s\n" "$1" "$2" "$3" >> "$RESULTS"
    fi
}

# check_program <key> <label> <src>
#
#   key     first field of the RESULTS line (a seed number, or a file stem)
#   label   what failure messages name the program by
#   src     path to the C source
#
# Returns 0 when every oracle agrees, 1 on any disagreement, 2 when the program
# does not terminate under the reference compiler either.
check_program() {
    _key="$1"
    _label="$2"
    _src="$3"

    want="$(sed -n 's|^// OUTPUT: ||p' "$_src")"

    # A reference answer, when the reference compiler will take the program.
    #
    # If the reference build runs but does not finish, the program itself does
    # not terminate and no oracle can say anything about it: blitz will time out
    # too, and blaming blitz for that is a false positive. Skip it and say so,
    # because it means the generator emitted something it should not have --
    # that is how two nonterminating programs were chased as miscompiles.
    wantc=""
    if command -v "$CC" > /dev/null 2>&1 \
        && "$CC" -w -O0 -ffp-contract=off -x c "$_src" -o "$WORK/ref" 2>/dev/null; then
        # `|| refst=$?` rather than a bare assignment: `set -e` is on and the
        # program under test may legitimately exit nonzero.
        refst=0
        refout="$(timeout 20 "$WORK/ref" 2>/dev/null)" || refst=$?
        if [ "$refst" -eq 124 ]; then
            printf "\nSKIP %s: does not terminate under %s either -- generator bug\n  %s\n" \
                "$_label" "$CC" "$_src"
            return 2
        fi
        if [ "$refst" -eq 0 ]; then
            wantc="$refout"
        fi
    fi

    # Check each level on its own, and keep going after a failure.
    #
    # These used to short-circuit: -O1 was compiled first and a failure there
    # skipped the program entirely. A level that cannot allocate registers then
    # hid whatever the other level did, and three -O0 miscompiles sat behind -O1
    # compile errors for a whole session. A compile error at one level says
    # nothing about the other.
    prog_failed=0
    o0_out=""; o0_ok=0
    o1_out=""; o1_ok=0
    for level in -O0 -O1; do
        # Compilation is under a timeout of its own. A compiler that loops
        # forever otherwise absorbs the entire run: one hang in the parallel-copy
        # sequentializer ate a 40-program sweep before anyone noticed the harness
        # was not merely slow. A hang is a finding, and reported as one.
        if ! timeout "$COMPILE_TIMEOUT" "$TINYC" "$_src" "$level" -o "$WORK/o" \
            > "$WORK/log" 2>&1; then
            st=$?
            prog_failed=1
            if [ "$st" -eq 124 ]; then
                printf "\nFAIL %s: blitz %s HUNG (over %ss)\n  %s\n" \
                    "$_label" "$level" "$COMPILE_TIMEOUT" "$_src"
                record "$_key" "$level" "fail hang"
            else
                printf "\nFAIL %s: blitz %s did not compile\n  %s\n" "$_label" "$level" "$_src"
                head -2 "$WORK/log" | sed 's/^/  /'
                record "$_key" "$level" "fail no-compile"
            fi
            continue
        fi
        # `set -e` is on, and a program under test may legitimately exit
        # nonzero -- that is a finding, not a reason to abandon the run. The
        # original code had the same shape and aborted the whole harness the
        # first time a compiled program returned nonzero.
        if out="$(timeout 20 "$WORK/o" 2>/dev/null)"; then st=0; else st=$?; fi
        if [ "$level" = "-O0" ]; then o0_out="$out"; o0_ok=1; else o1_out="$out"; o1_ok=1; fi
        if [ "$st" -ne 0 ]; then
            prog_failed=1
            printf "\nFAIL %s: blitz %s exited %s\n  %s\n" "$_label" "$level" "$st" "$_src"
            record "$_key" "$level" "fail exit-nonzero"
            continue
        fi
        if [ -n "$want" ] && [ "$out" != "$want" ]; then
            prog_failed=1
            printf "\nFAIL %s: blitz %s printed %s, generator predicted %s\n  %s\n" \
                "$_label" "$level" "$out" "$want" "$_src"
            record "$_key" "$level" "fail wrong-predicted"
            continue
        fi
        if [ -n "$wantc" ] && [ "$out" != "$wantc" ]; then
            prog_failed=1
            printf "\nFAIL %s: blitz %s printed %s, %s printed %s\n  %s\n" \
                "$_label" "$level" "$out" "$CC" "$wantc" "$_src"
            record "$_key" "$level" "fail wrong-cc"
            continue
        fi
        record "$_key" "$level" "pass"
    done

    # -O0-vs-O1 self-consistency, when both levels produced a program. This
    # catches a pass that changes behavior even where no oracle disagrees.
    if [ "$o0_ok" = 1 ] && [ "$o1_ok" = 1 ] && [ "$o0_out" != "$o1_out" ]; then
        prog_failed=1
        printf "\nFAIL %s: -O0 printed %s, -O1 printed %s\n  %s\n" \
            "$_label" "$o0_out" "$o1_out" "$_src"
        record "$_key" both "fail levels-disagree"
    fi

    return "$prog_failed"
}
