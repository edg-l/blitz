#!/usr/bin/env python3
"""Find which terms of a generated program's sum blitz gets wrong.

Adds a constant to one initialiser at a time and compares how blitz's answer
moves against how the reference compiler's answer moves. The reference validates
every probe: where the two deltas differ, that term's value is wrong in blitz,
and where they agree the term is fine however surprising the delta looks.

This localises a wrong value without reading any disassembly. On a 64-line
program with 365 virtual registers it named the faulty terms in one run, which
reading the emitted code had not.

    perturb.py INPUT.c [--opt=-O0] [--cc cc] [--blitz PATH] [--delta 1000]

Both compilers must accept the program and produce a single integer on stdout.
A term whose delta is `n/a` means one of them failed to compile or run the
perturbed variant; that is a probe failure, not a finding.

Nothing here assumes the generator's `// OUTPUT:` prediction: it stops being
valid the moment an initialiser changes, so the reference compiler is the oracle.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile


def build_and_run(compiler_cmd, binary):
    """Compile and run, returning stdout stripped, or None if either step fails."""
    if subprocess.run(compiler_cmd, capture_output=True).returncode != 0:
        return None
    proc = subprocess.run([binary], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


class Oracle:
    def __init__(self, cc, blitz, opt):
        self.cc, self.blitz, self.opt = cc, blitz, opt
        self.dir = tempfile.mkdtemp(prefix="perturb_")

    def answers(self, text, tag):
        src = os.path.join(self.dir, f"{tag}.c")
        with open(src, "w") as f:
            f.write(text)
        cc_bin = os.path.join(self.dir, f"{tag}.cc")
        bz_bin = os.path.join(self.dir, f"{tag}.bz")
        return (
            build_and_run([self.cc, "-O0", "-w", src, "-o", cc_bin], cc_bin),
            build_and_run([self.blitz, self.opt, src, "-o", bz_bin], bz_bin),
        )


def initialisers(src):
    """Every `int vN = C;` and `arr[N] = C;` in source order, as (label, match).

    Commented-out lines are skipped: a findings header that quotes an
    initialiser would otherwise be probed as if it were code.
    """
    comments = [
        (m.start(), m.end()) for m in re.finditer(r"^[ \t]*//.*$", src, re.MULTILINE)
    ]

    def in_comment(pos):
        return any(start <= pos < end for start, end in comments)

    found = []
    for pattern, group in ((r"\bint (v\d+) = (-?\d+);", 1), (r"\b(arr\[\d+\]) = (-?\d+);", 1)):
        for m in re.finditer(pattern, src):
            if not in_comment(m.start()):
                found.append((m.group(group), m))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--opt", default="-O0", help="blitz opt level (default -O0)")
    ap.add_argument("--cc", default="cc")
    ap.add_argument("--blitz", default="./target/debug/tinyc")
    ap.add_argument("--delta", type=int, default=1000)
    args = ap.parse_args()

    with open(args.input) as f:
        src = f.read()
    oracle = Oracle(args.cc, args.blitz, args.opt)

    base_cc, base_bz = oracle.answers(src, "base")
    if base_cc is None:
        sys.exit(f"{args.cc} does not accept {args.input}")
    if base_bz is None:
        sys.exit(f"blitz does not accept {args.input} at {args.opt}")
    print(f"base: cc={base_cc} blitz={base_bz}")
    if base_cc == base_bz:
        print("they agree; nothing to localise")

    def delta(new, base):
        if new is None:
            return None
        try:
            return int(new) - int(base)
        except ValueError:
            return None

    mismatches = []
    for label, m in initialisers(src):
        old = int(m.group(2))
        probe = src[: m.start()] + m.group(0).replace(
            f"= {old};", f"= {old + args.delta};"
        ) + src[m.end():]
        cc_out, bz_out = oracle.answers(probe, label.replace("[", "").replace("]", ""))
        d_cc, d_bz = delta(cc_out, base_cc), delta(bz_out, base_bz)
        bad = d_cc is None or d_bz is None or d_cc != d_bz
        if bad and d_cc is not None and d_bz is not None:
            mismatches.append(label)
        print(
            f"  {label:10s} cc delta={'n/a' if d_cc is None else d_cc:>8}"
            f"  blitz delta={'n/a' if d_bz is None else d_bz:>8}"
            f"{'  <== WRONG' if bad else ''}"
        )

    if mismatches:
        print(f"\nwrong terms: {', '.join(mismatches)}")
    else:
        print("\nno term's delta disagrees")


if __name__ == "__main__":
    main()
