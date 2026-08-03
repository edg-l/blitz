#!/usr/bin/env python3
"""Read what the emitted code actually holds, from the unmodified binary.

Every other probe changes the program. Near the allocator's limit that moves the
bug: an extra instruction shifts register pressure, and even a changed constant
folds differently downstream, so a term that reads as wrong under perturbation can
be right in the program as written. This script adds nothing -- it breaks at an
address in the binary blitz already produced and prints registers and frame slots.

Two modes.

`--sum-chain` walks the trailing `add` chain that a generated program's final
printf consumes and prints, for each add, the term being added and the running
total before it. The first line whose running total is not what the program should
have computed names the first wrong term, which is the question perturbation can
only answer approximately:

    read_frame.py prog.c --sum-chain

`--at` breaks at one or more addresses and prints whatever is asked for. The
addresses are the ones `--emit-asm` prints, which restart at zero for each
function, so they are used as `main+<addr>` directly:

    read_frame.py prog.c --at 0x6a0 --slot 0x290 --reg esi

Slots are frame displacements off RSP, exactly as the disassembly and the slot
traffic dump (`BLITZ_DEBUG=slots`) name them.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

# `   6a0:\t48 8b bc 24 90 02 00 \tmov    rdi,QWORD PTR [rsp+0x290]`
ASM_LINE = re.compile(r"^\s*([0-9a-f]+):\t[0-9a-f ]+\t(.*)$")
ADD_REG_REG = re.compile(r"^add\s+([a-z0-9]+),([a-z0-9]+)$")


def emit_asm(blitz, src, opt, func):
    """The disassembly of one function, as (addr, text) pairs."""
    proc = subprocess.run([blitz, opt, "--emit-asm", src], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"blitz could not compile {src} at {opt}:\n{proc.stdout}{proc.stderr}")
    lines = proc.stdout.splitlines()
    try:
        start = lines.index(f"# {func}")
    except ValueError:
        sys.exit(f"no `# {func}` section in --emit-asm output")
    out = []
    for line in lines[start + 1:]:
        if line.startswith("# "):
            break
        m = ASM_LINE.match(line)
        if m:
            out.append((int(m.group(1), 16), m.group(2).strip()))
    return out


def sum_chain(asm):
    """The trailing register-register adds before the last call, in order.

    Returns [(addr, total_reg, term_reg)]. The generated sum is left-associative,
    so these are the terms in source order.
    """
    calls = [i for i, (_, text) in enumerate(asm) if text.startswith("call")]
    if not calls:
        return []
    last = calls[-1]
    chain = []
    for addr, text in asm[:last]:
        m = ADD_REG_REG.match(text)
        if m:
            chain.append((addr, m.group(1), m.group(2)))
    return chain


def gdb_read(binary, func, requests):
    """Run the binary under gdb, reading values at breakpoints.

    `requests` is [(addr, [expr, ...])]. Returns {addr: [value, ...]}, with a
    value of None where gdb could not read it. One process, one run: each
    breakpoint fires in address order and the commands are attached to it, so a
    program with a loop reports the first time through.
    """
    script = []
    for addr, exprs in requests:
        script += ["-ex", f"break *({func}+{addr:#x})"]
    script += ["-ex", "run"]
    # After the run stops at the first breakpoint, step through them in order.
    for i, (addr, exprs) in enumerate(requests):
        script += ["-ex", f"printf \"@{addr:#x}\\n\""]
        for e in exprs:
            script += ["-ex", f"printf \"{e} = %d\\n\", {e}"]
        if i + 1 < len(requests):
            script += ["-ex", "continue"]
    proc = subprocess.run(
        ["gdb", "-batch", "-nx", *script, binary],
        capture_output=True,
        text=True,
    )
    values, current = {}, None
    for line in proc.stdout.splitlines():
        if line.startswith("@"):
            current = int(line[1:], 16)
            values[current] = []
        elif " = " in line and current is not None:
            values[current].append(line.split(" = ", 1)[1].strip())
    if not values:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit("gdb produced no readings; is the address mid-instruction?")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--opt", default="-O0")
    ap.add_argument("--func", default="main")
    ap.add_argument("--sum-chain", action="store_true", help="walk the final printf's add chain")
    ap.add_argument("--at", action="append", default=[], help="address from --emit-asm, hex")
    ap.add_argument("--slot", action="append", default=[], help="frame displacement off RSP, hex")
    ap.add_argument("--reg", action="append", default=[], help="register name, e.g. esi")
    ap.add_argument(
        "--blitz", default=os.path.join(os.path.dirname(__file__), "../../target/debug/tinyc")
    )
    args = ap.parse_args()

    blitz = os.path.abspath(args.blitz)
    asm = emit_asm(blitz, args.input, args.opt, args.func)

    with tempfile.TemporaryDirectory(prefix="read_frame_") as work:
        binary = os.path.join(work, "prog")
        if subprocess.run(
            [blitz, args.opt, args.input, "-o", binary], capture_output=True
        ).returncode != 0:
            sys.exit(f"blitz could not compile {args.input} at {args.opt}")

        if args.sum_chain:
            chain = sum_chain(asm)
            if not chain:
                sys.exit("no register-register add chain before the last call")
            requests = [
                (addr, [f"${total}", f"${term}"]) for addr, total, term in chain
            ]
            values = gdb_read(binary, args.func, requests)
            print(f"{'term':>5}  {'addr':>8}  running total  term added")
            for i, (addr, total, term) in enumerate(chain):
                got = values.get(addr, [])
                have_total = got[0] if len(got) > 0 else "?"
                have_term = got[1] if len(got) > 1 else "?"
                print(
                    f"{i:>5}  {addr:#8x}  {have_total:>13}  {have_term:>10}"
                    f"   ({total} += {term})"
                )
            print(
                "\nThe running total is the sum of every earlier term. The first row"
                "\nwhose total is not what the program should have reached is the"
                "\nfirst wrong term."
            )
            return

        if not args.at:
            sys.exit("nothing to do: pass --sum-chain or --at")
        exprs = []
        for s in args.slot:
            disp = int(s, 16)
            exprs.append(f"*(long *)($rsp+{disp:#x})")
        for r in args.reg:
            exprs.append(f"${r}")
        if not exprs:
            sys.exit("pass at least one --slot or --reg")
        requests = [(int(a, 16), exprs) for a in args.at]
        values = gdb_read(binary, args.func, requests)
        labels = [f"[rsp+{int(s, 16):#x}]" for s in args.slot] + [f"%{r}" for r in args.reg]
        for addr, _ in requests:
            got = values.get(addr, [])
            text = dict(asm).get(addr, "?")
            print(f"{addr:#x}  {text}")
            for label, value in zip(labels, got):
                print(f"    {label:>16} = {value}")


if __name__ == "__main__":
    main()
