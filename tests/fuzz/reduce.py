#!/usr/bin/env python3
"""Line-based delta debugger for a C program that blitz miscompiles.

Repeatedly deletes lines and keeps any deletion that leaves the program still
failing, until nothing more can go. A candidate counts as still failing when

  * the reference compiler accepts it,
  * `cc -O0` and `cc -O2` agree, which is the cheap check that the reduction has
    not introduced undefined behaviour and started comparing against noise,
  * blitz still compiles it, and
  * blitz still disagrees with the reference.

The "blitz still compiles it" condition is what keeps the reduction on the bug
it started from. Without it the search happily drifts onto any internal panic or
"cannot allocate registers" it can reach, which is a different bug and usually an
easier one to reach -- the first run of this script turned a wrong-value bug into
a missing-return crash in nine steps. Pass --want-compile-failure to reduce a
crash on purpose.

Usage: reduce.py INPUT.c [-o OUT.c] [--opt -O0] [--cc cc]

The generator's own `// OUTPUT:` prediction stops being valid the moment a line
is deleted, so the reference compiler is the oracle here.
"""
import argparse, os, subprocess, sys, tempfile

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)

class Oracle:
    def __init__(self, cc, blitz, opt, want_compile_failure=False):
        self.cc, self.blitz, self.opt = cc, blitz, opt
        self.want_compile_failure = want_compile_failure
        self.ub_check = False
        self.dir = tempfile.mkdtemp(prefix="reduce_")

    def _build_run(self, compiler_cmd, binary):
        if run(compiler_cmd).returncode != 0:
            return None
        p = run([binary])
        return (p.returncode, p.stdout)

    # Reductions that fall off the end of a non-void function, or read a local
    # the reducer just stopped initialising, are undefined -- and both are
    # exactly what deleting lines produces. `cc -O0` and `cc -O2` agreeing does
    # not catch them: an uninitialised read usually lands on the same garbage at
    # both levels, and a missing return usually leaves the same register alone.
    # Two reductions were chased as miscompiles before this check existed, and a
    # third the first time this script was pointed at a program with an array.
    #
    # These warnings are conservative in the right direction: a candidate the
    # compiler is unsure about is one the oracle cannot vouch for either.
    # The generator's own `extern int printf(char*, int)` conflicts with the
    # builtin, which is unrelated to anything a reduction can introduce.
    UB_WARNINGS = [
        "-Wno-builtin-declaration-mismatch",
        "-Werror=return-type",
        "-Werror=uninitialized",
        "-Werror=maybe-uninitialized",
    ]

    def _is_well_defined(self, src):
        # The build commands pass -w, which silences everything, so ask
        # separately and only about the constructs deleting a line introduces.
        #
        # This must be a real -O2 compile, not -fsyntax-only: reaching the end
        # of a non-void function and reading an uninitialised local are both
        # found by the optimiser's dataflow, and neither is reported without it.
        # With -fsyntax-only this check passed a reduction that had lost two
        # `return`s and seven of eight array initialisers.
        if not self.ub_check:
            return True
        cmd = [self.cc, "-O2", "-c", "-o", os.devnull] + self.UB_WARNINGS + [src]
        return run(cmd).returncode == 0

    def calibrate_ub_check(self, text):
        """Enable the UB check only if the starting program passes it.

        The flag set is not portable -- gcc and clang disagree about which
        -W names exist -- and a compiler that rejects an option rejects the
        whole invocation, which would read as "every candidate is undefined"
        and reduce nothing. Anchoring on the input means the check can only
        ever reject candidates the original did not have.
        """
        src = os.path.join(self.dir, "cal.c")
        with open(src, "w") as f:
            f.write(text)
        self.ub_check = True
        if not self._is_well_defined(src):
            self.ub_check = False
            print("note: UB warning check unavailable, reductions may introduce UB",
                  file=sys.stderr)

    def fails(self, text):
        src = os.path.join(self.dir, "c.c")
        with open(src, "w") as f:
            f.write(text)
        if not self._is_well_defined(src):
            return False
        ref0 = os.path.join(self.dir, "ref0")
        ref2 = os.path.join(self.dir, "ref2")
        r0 = self._build_run([self.cc, "-w", "-O0", src, "-o", ref0], ref0)
        if r0 is None:
            return False
        r2 = self._build_run([self.cc, "-w", "-O2", src, "-o", ref2], ref2)
        # Disagreement between the reference's own levels means the reduction
        # has introduced undefined behaviour; the program no longer has one
        # right answer to hold blitz to.
        if r2 is None or r0 != r2:
            return False
        out = os.path.join(self.dir, "b")
        b = self._build_run([self.blitz, self.opt, src, "-o", out], out)
        if b is None:
            return self.want_compile_failure
        return not self.want_compile_failure and b != r0

def reduce_lines(lines, oracle):
    n = len(lines)
    chunk = max(n // 2, 1)
    while chunk >= 1:
        i = 0
        while i < len(lines):
            candidate = lines[:i] + lines[i + chunk:]
            if oracle.fails("".join(candidate)):
                lines = candidate
            else:
                i += chunk
        if chunk == 1:
            break
        chunk = max(chunk // 2, 1)
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--out")
    ap.add_argument("--opt", default="-O0")
    ap.add_argument("--cc", default="cc")
    ap.add_argument(
        "--want-compile-failure",
        action="store_true",
        help="reduce a blitz crash or compile error instead of a wrong answer",
    )
    ap.add_argument(
        "--blitz",
        default=os.path.join(os.path.dirname(__file__), "../../target/debug/tinyc"),
    )
    args = ap.parse_args()

    text = open(args.input).read()
    oracle = Oracle(
        args.cc, os.path.abspath(args.blitz), args.opt, args.want_compile_failure
    )
    oracle.calibrate_ub_check(text)
    if not oracle.fails(text):
        print("input does not fail the oracle; nothing to reduce", file=sys.stderr)
        return 1

    lines = text.splitlines(keepends=True)
    print(f"start: {len(lines)} lines", file=sys.stderr)
    lines = reduce_lines(lines, oracle)
    result = "".join(lines)
    print(f"done:  {len(lines)} lines", file=sys.stderr)
    if args.out:
        open(args.out, "w").write(result)
    else:
        sys.stdout.write(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
