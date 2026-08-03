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


def gdb_read(binary, func, requests, limit=200):
    """Run the binary under gdb, reading values at breakpoints.

    `requests` is [(addr, [expr, ...])]. Returns {addr: [[value, ...], ...]} --
    one inner list per time that address was reached, in execution order.

    Every reading is labelled with the `$pc` gdb actually stopped at, never with
    the order the addresses were asked for. Attaching the reads to a fixed
    sequence of `continue`s silently mislabels everything the moment an address
    sits in a loop or the addresses are reached out of order, which is the normal
    case and cost one wrong diagnosis before this was fixed.
    """
    lines = ["set confirm off", "set pagination off", "set height 0"]
    for addr, exprs in requests:
        lines.append(f"break *({func}+{addr:#x})")
        lines.append("commands")
        lines.append("silent")
        lines.append('printf "@%p\\n", $pc')
        for e in exprs:
            lines.append(f'printf "{e} = %d\\n", {e}')
        lines.append("continue")
        lines.append("end")
    lines.append("run")
    with tempfile.NamedTemporaryFile("w", suffix=".gdb", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        script_path = f.name
    proc = subprocess.run(
        ["gdb", "-batch", "-nx", "-x", script_path, binary],
        capture_output=True,
        text=True,
    )
    os.unlink(script_path)

    # `$pc` is absolute; map it back to the offsets the caller asked about.
    base = None
    m = re.search(r"Breakpoint 1 at (0x[0-9a-f]+)", proc.stdout)
    if m and requests:
        base = int(m.group(1), 16) - requests[0][0]

    values, current, count = {}, None, 0
    for line in proc.stdout.splitlines():
        if line.startswith("@"):
            pc = int(line[1:], 16)
            current = pc - base if base is not None else pc
            values.setdefault(current, [])
            values[current].append([])
            count += 1
            if count > limit:
                break
        elif " = " in line and current is not None and values.get(current):
            values[current][-1].append(line.split(" = ", 1)[1].strip())
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
                # First time through; the sum runs once, but a guard above it may
                # not, so this is explicit rather than assumed.
                occurrences = values.get(addr, [])
                got = occurrences[0] if occurrences else []
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
            occurrences = values.get(addr, [])
            text = dict(asm).get(addr, "?")
            print(f"{addr:#x}  {text}")
            if not occurrences:
                print("    never reached")
                continue
            for n, got in enumerate(occurrences):
                prefix = f"  #{n}" if len(occurrences) > 1 else "    "
                for label, value in zip(labels, got):
                    print(f"{prefix} {label:>16} = {value}")


if __name__ == "__main__":
    main()
