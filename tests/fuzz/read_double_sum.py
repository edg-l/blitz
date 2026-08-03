#!/usr/bin/env python3
"""Read the value a compiler actually computes for an expression, without adding code.

`gen_c.py` ends every program with a guard of the form

    if ((((((d0 + d1) + d2) + d3) + d4) + d5) != <K>) { return 3; }

so a wrong floating-point value shows up only as `exit 3` -- no number, nothing to
compare, and no way to tell which of the six doubles went wrong.

The obvious probes all fail on the programs that need them most. Printing the
doubles adds a call per value and pushes the function past what the allocator can
colour; converting the sum to an int and printing it does the same; adding one
equality guard per double changes register allocation enough to move the fault
somewhere else. In one measured case six guards blamed d0 while one guard at a
time blamed d4.

This replaces the guard with

    if (<expr> > T) { return 4; }

which is the same shape as the guard it replaces -- one compare, one branch -- and
bisects T. The exit code answers "is <expr> above T", and 60 iterations pin the
value. Agreement between -O0 and -O1 is the check that the probe is not itself
what is being measured.

Usage:
  read_double_sum.py PROG.c                      # the sum, per compiler
  read_double_sum.py PROG.c -e d4 -e 'd2 * d5'   # named expressions
  read_double_sum.py PROG.c -e d2 --before 'd4 = (d4 + (d2 * d5));'

`--before` moves the probe to just above a statement instead of the end, which is
how to see an operand as some particular statement sees it. The original guard is
removed in that case so the instruction count stays put.
"""
import argparse
import re
import subprocess
import sys
import tempfile
import os

GUARD = re.compile(
    r"^\s*if \(\(\(\(\(\(d0 \+ d1\) \+ d2\) \+ d3\) \+ d4\) \+ d5\) != [^)]*\) \{ return 3; \}$",
    re.M,
)
SUM = "(((((d0 + d1) + d2) + d3) + d4) + d5)"


def build_and_run(text, compiler, level, blitz, cc, workdir):
    src = os.path.join(workdir, "probe.c")
    with open(src, "w") as f:
        f.write(text)
    exe = os.path.join(workdir, "probe.exe")
    cmd = [cc, "-w", "-O0", src, "-o", exe] if compiler == "cc" else [blitz, level, src, "-o", exe]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return None
    return subprocess.run([exe], capture_output=True).returncode


def probe_text(base, expr, thresh, before):
    guarded = f"    if ({expr} > {thresh}) {{ return 4; }}"
    if before is None:
        return GUARD.sub(guarded, base, count=1)
    if before not in base:
        sys.exit(f"--before statement not found verbatim: {before!r}")
    return GUARD.sub("", base, count=1).replace(before, f"{guarded}\n{before}", 1)


def read(base, expr, compiler, level, before, blitz, cc, workdir):
    def above(t):
        code = build_and_run(
            probe_text(base, expr, t, before), compiler, level, blitz, cc, workdir
        )
        return None if code is None else code == 4

    lo, hi = -1e13, 1e13
    top = above(hi)
    if top is None:
        return "did not build"
    if top:
        return "> 1e13"
    if not above(lo):
        return "< -1e13"
    for _ in range(60):
        mid = (lo + hi) / 2
        r = above(mid)
        if r is None:
            return "build failed mid-bisection"
        if r:
            lo = mid
        else:
            hi = mid
    # The residual is bisection noise, not a real fraction.
    return round(hi, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-e", "--expr", action="append", help="expression to read (default: the sum)")
    ap.add_argument("--before", help="place the probe above this statement, verbatim")
    ap.add_argument("--cc", default="cc")
    ap.add_argument(
        "--blitz", default=os.path.join(os.path.dirname(__file__), "../../target/debug/tinyc")
    )
    args = ap.parse_args()

    base = open(args.input).read()
    if not GUARD.search(base):
        sys.exit("no generated double-sum guard in this file; nothing to replace")
    blitz = os.path.abspath(args.blitz)
    exprs = args.expr or [SUM]

    with tempfile.TemporaryDirectory(prefix="read_double_") as workdir:
        for expr in exprs:
            vals = [
                read(base, expr, c, lvl, args.before, blitz, args.cc, workdir)
                for c, lvl in [("cc", "-O0"), ("blitz", "-O0"), ("blitz", "-O1")]
            ]
            same = vals[0] == vals[1] == vals[2]
            print(
                f"{expr:24} cc={vals[0]}  blitz -O0={vals[1]}  blitz -O1={vals[2]}"
                f"{'' if same else '   <-- DIFFERS'}"
            )


main()
