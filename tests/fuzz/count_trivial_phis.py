#!/usr/bin/env python3
"""How many of a program's block parameters are redundant phis?

`phi(x, x, ..., x) -> x`, self-references ignored (Braun et al.), applied to a
fixpoint since removing one trivial phi can leave another with a single source.
Reads `--emit-ir` and answers per block, so it costs nothing and needs no compiler
change to run.

    count_trivial_phis.py prog.c -O1

Measured 85-94% redundant across the generated corpus, with one loop header
carrying 28 parameters of which 4 were real. That is the size of the prize for
`src/compile/phi_simplify.rs`, and the reason to keep working on it: each redundant
parameter costs a register-to-register copy per incoming edge per iteration, a
place in the block's parameter clique, and -- once the splitter routes what cannot
be coloured -- a store and a reload per iteration for a value nothing reads.

This counts what the *rule* permits, not what is *sound* to remove: one e-class is
one expression, not one value. See the module doc of `phi_simplify.rs`.
"""
import re, sys, subprocess

def load(src, opt):
    out = subprocess.run(["/home/edgar/dev/blitz/target/release/tinyc", opt, "--emit-ir", src],
                         capture_output=True, text=True).stdout
    fn = out[out.index("function main"):] if "function main" in out else out
    params = {}   # (block, idx) -> class
    edges = []    # (target, [arg classes])
    types  = {}   # block -> [type strings]
    for line in fn.splitlines():
        m = re.match(r"\s*v(\d+) = block_param\(b(\d+), (\d+), (\w+)\)", line)
        if m:
            params[(int(m.group(2)), int(m.group(3)))] = int(m.group(1))
            continue
        m = re.match(r"\s*block(\d+)\((.*)\):", line)
        if m:
            types[int(m.group(1))] = [p.split(": ")[1] for p in m.group(2).split(", ")]
            continue
        for m in re.finditer(r"(?:jump|block) block(\d+)\(([^)]*)\)", line):
            args = m.group(2).strip()
            if args:
                edges.append((int(m.group(1)), [int(a.strip().lstrip("v")) for a in args.split(",")]))
    return params, edges, types

def analyse(params, edges, types):
    # A parameter with no block_param line has no class we can name; treat the
    # position as present but opaque (it cannot be proven trivial).
    incoming = {}
    for target, args in edges:
        for i, a in enumerate(args):
            incoming.setdefault((target, i), []).append(a)
    # union-find over "this param is really that value"
    alias = {}
    def find(x):
        while x in alias and alias[x] != x:
            x = alias[x]
        return x
    changed = True
    trivial = set()
    while changed:
        changed = False
        for (b, i), cls in params.items():
            if (b, i) in trivial:
                continue
            srcs = incoming.get((b, i))
            if not srcs:
                continue
            distinct = {find(s) for s in srcs} - {find(cls)}
            if len(distinct) == 1:
                alias[find(cls)] = distinct.pop()
                trivial.add((b, i))
                changed = True
    return trivial

for src, opt in [(sys.argv[1], sys.argv[2])]:
    params, edges, types = load(src, opt)
    trivial = analyse(params, edges, types)
    blocks = sorted({b for b, _ in params})
    print(f"{src} {opt}")
    print(f"{'block':>6} {'params':>7} {'with class':>11} {'redundant':>10} {'survive':>8}")
    tot_p = tot_r = 0
    for b in blocks:
        n_declared = len(types.get(b, []))
        withcls = sum(1 for (bb, _) in params if bb == b)
        red = sum(1 for (bb, _) in trivial if bb == b)
        if n_declared == 0:
            continue
        tot_p += n_declared; tot_r += red
        if n_declared >= 5:
            print(f"{b:>6} {n_declared:>7} {withcls:>11} {red:>10} {n_declared-red:>8}")
    print(f"TOTAL params {tot_p}, provably redundant {tot_r} ({100*tot_r//max(tot_p,1)}%)")
