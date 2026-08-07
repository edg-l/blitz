#!/usr/bin/env python3
"""Generate random C programs, free of undefined behavior, aimed at the parts of
the backend the hand-written corpus does not reach.

Why UB-freedom is the central constraint
----------------------------------------
The reference-compiler oracle in tests/lit/run_diff.sh is only useful if a
mismatch means a compiler bug. `++c + ++c` is legal C whose value gcc and blitz
may each compute differently, so a generator that can emit UB manufactures
mismatches that are nobody's fault. Every construct here makes a hazard
unreachable rather than unlikely:

  unsequenced side effects   side effects are statements; expressions are pure
  evaluation order           call arguments are pure, so order cannot matter
  signed overflow            the program is simulated as it is built and
                             rejected if any value leaves SAFE_RANGE
  divide by zero             divisors are nonzero constants
  shift out of range         shift counts are constants in 0..4, operand >= 0
  out-of-bounds indexing     indices are masked with & (ARRAY_SIZE-1), which
                             for a power-of-two size lands in bounds for every
                             input including negatives (unlike %, which keeps
                             the sign of its left operand)
  uninitialized reads        every declaration has an initializer
  non-terminating loops      trip counts are constants
  float rounding             doubles hold integral values and use only + - *,
                             so every result is exact in binary64 and no
                             reference compiler can disagree (this also makes
                             FP contraction irrelevant)

Because generation interprets the program, the expected result is known, so the
file carries `// OUTPUT:` and `// EXIT:` directives and is self-checking under
run_tests.sh even with no reference compiler.

What it targets
---------------
Measured against tests/lit, which has at most 7 parameters on any function and
at most ~13 live locals:

  args      functions of 7-12 parameters, crossing the 6-GPR / 8-XMM argument
            register boundary into stack arguments, whose callee read side
            ROADMAP lists as unverified
  mixed     interleaved int and double parameters, so the GPR and XMM argument
            indices advance independently -- the shape that produced the
            `cvtsi2sd xmm0, rsp` bug
  pressure  more simultaneously-live values than there are registers, all live
            across calls, which is where the XMM splitter bug lived

Usage:
    python3 gen_c.py --seed 42 [--shape mixed] [--out prog.c]
"""

import argparse
import random
import sys

SAFE_RANGE = 30000  # keeps every int product inside int32
FP_LIMIT = 1 << 40  # keeps every double exactly representable
ARRAY_SIZE = 8


class Reject(Exception):
    """Raised during simulation when an operation would not be well-defined."""


class Nonterminating(Reject):
    """Raised when a loop exceeds `LOOP_STEP_CAP` iterations.

    A program that does not finish is as useless as one with undefined
    behavior: every compiler hangs on it, so no oracle can disagree, and the
    harness has nothing to compare. Rejecting it here is the same contract as
    `Reject` -- the seed is reported as ungeneratable and skipped.
    """


# Generated loops have trip counts of 1..6 and nothing writes to a counter, so
# this is orders of magnitude above anything legitimate.
LOOP_STEP_CAP = 10_000


def ck(v):
    if not -SAFE_RANGE <= v <= SAFE_RANGE:
        raise Reject(v)
    return v


def ckf(v):
    # Integral and bounded => exact in binary64 on every conforming compiler.
    if v != int(v) or abs(v) > FP_LIMIT:
        raise Reject(v)
    return v


def c_div(a, b):
    """C division truncates toward zero; Python's // floors."""
    if b == 0:
        raise Reject("div0")
    return int(a / b)


# ── Expressions (always pure) ────────────────────────────────────────────────


class Const:
    def __init__(self, v):
        self.v = v

    def render(self):
        return str(self.v)

    def eval(self, env):
        return self.v


class FConst:
    def __init__(self, v):
        self.v = float(v)

    def render(self):
        return f"{self.v:.1f}"

    def eval(self, env):
        return self.v


class Var:
    def __init__(self, name):
        self.name = name

    def render(self):
        return self.name

    def eval(self, env):
        return env[self.name]


class Index:
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx

    def render(self):
        return f"{self.name}[({self.idx.render()}) & {ARRAY_SIZE - 1}]"

    def eval(self, env):
        # A power-of-two mask is in bounds for every input, negatives included,
        # and Python's & on negative ints matches two's complement as C does.
        return env[self.name][self.idx.eval(env) & (ARRAY_SIZE - 1)]


class Bin:
    def __init__(self, op, l, r):
        self.op = op
        self.l = l
        self.r = r

    def render(self):
        return f"({self.l.render()} {self.op} {self.r.render()})"

    def eval(self, env):
        a = self.l.eval(env)
        b = self.r.eval(env)
        if self.op == "+":
            return ck(a + b)
        if self.op == "-":
            return ck(a - b)
        if self.op == "*":
            return ck(a * b)
        if self.op == "/":
            return ck(c_div(a, b))
        if self.op == "%":
            return ck(a - c_div(a, b) * b)
        if self.op == "<<":
            if a < 0 or not 0 <= b <= 31:
                raise Reject("shl")
            return ck(a << b)
        if self.op == ">>":
            if a < 0 or not 0 <= b <= 31:
                raise Reject("shr")
            return a >> b
        if self.op in ("&", "|", "^"):
            # Well-defined on two's complement, which gcc, clang and blitz all
            # implement on x86-64; Python's operators agree bit for bit.
            return ck({"&": a & b, "|": a | b, "^": a ^ b}[self.op])
        return int(
            {
                "<": a < b, ">": a > b, "<=": a <= b,
                ">=": a >= b, "==": a == b, "!=": a != b,
            }[self.op]
        )


class FBin:
    """Double arithmetic restricted to + - * over integral values, so exact."""

    def __init__(self, op, l, r):
        self.op = op
        self.l = l
        self.r = r

    def render(self):
        return f"({self.l.render()} {self.op} {self.r.render()})"

    def eval(self, env):
        a = self.l.eval(env)
        b = self.r.eval(env)
        return ckf({"+": a + b, "-": a - b, "*": a * b}[self.op])


class IntToF:
    """An int used where a double is wanted: the implicit conversion is exact."""

    def __init__(self, e):
        self.e = e

    def render(self):
        return self.e.render()

    def eval(self, env):
        return float(self.e.eval(env))


class Call:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def render(self):
        return f"{self.fn.name}({', '.join(a.render() for a in self.args)})"

    def eval(self, env):
        return self.fn.call([a.eval(env) for a in self.args])


# ── Statements ───────────────────────────────────────────────────────────────


class Decl:
    def __init__(self, ty, name, expr):
        self.ty = ty
        self.name = name
        self.expr = expr

    def render(self, ind):
        return f"{ind}{self.ty} {self.name} = {self.expr.render()};"

    def exec(self, env):
        env[self.name] = self.expr.eval(env)


class Assign:
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

    def render(self, ind):
        return f"{ind}{self.name} = {self.expr.render()};"

    def exec(self, env):
        env[self.name] = self.expr.eval(env)


class ArrAssign:
    def __init__(self, name, idx, expr):
        self.name = name
        self.idx = idx
        self.expr = expr

    def render(self, ind):
        return f"{ind}{self.name}[({self.idx.render()}) & {ARRAY_SIZE - 1}] = {self.expr.render()};"

    def exec(self, env):
        env[self.name][self.idx.eval(env) & (ARRAY_SIZE - 1)] = self.expr.eval(env)


class If:
    def __init__(self, cond, then, els):
        self.cond = cond
        self.then = then
        self.els = els

    def render(self, ind):
        out = [f"{ind}if ({self.cond.render()}) {{"]
        out += [s.render(ind + "    ") for s in self.then]
        out.append(f"{ind}}} else {{")
        out += [s.render(ind + "    ") for s in self.els]
        out.append(f"{ind}}}")
        return "\n".join(out)

    def exec(self, env):
        for s in self.then if self.cond.eval(env) else self.els:
            s.exec(env)


class For:
    def __init__(self, var, trips, body):
        self.var = var
        self.trips = trips
        self.body = body

    def render(self, ind):
        out = [f"{ind}for (int {self.var} = 0; {self.var} < {self.trips}; {self.var}++) {{"]
        out += [s.render(ind + "    ") for s in self.body]
        out.append(f"{ind}}}")
        return "\n".join(out)

    def exec(self, env):
        # C semantics, not `range(trips)`: initialise once, test, run the body,
        # increment whatever the body left behind. The old version re-assigned
        # the counter at the top of every iteration and ran exactly `trips`
        # times, so it silently disagreed with the emitted C the moment a body
        # statement wrote to the counter -- `i24 = p2` with p2 == 0 never
        # terminates in C while the interpreter happily predicted an answer.
        # Loop counters are no longer assignment targets, so the two agree; this
        # models C anyway, because an interpreter that cannot express the
        # divergence cannot warn about it either.
        #
        # A loop variable name may repeat in a nested loop, where C shadows the
        # outer one and restores it at the end of the inner scope. `env` is flat,
        # so save and restore by hand.
        missing = object()
        saved = env.get(self.var, missing)
        env[self.var] = 0
        steps = 0
        while env[self.var] < self.trips:
            for s in self.body:
                s.exec(env)
            env[self.var] = ck(env[self.var] + 1)
            steps += 1
            if steps > LOOP_STEP_CAP:
                raise Nonterminating(
                    f"loop on {self.var} ran {steps} iterations for trips={self.trips}"
                )
        if saved is missing:
            env.pop(self.var, None)
        else:
            env[self.var] = saved


class Func:
    """A function whose parameters may mix int and double.

    Mixed signatures are the point: SysV advances the GPR and XMM argument
    indices independently, so `f(int, double, int, double, ...)` puts arguments
    in a different register order than either all-int or all-double, and past
    six ints or eight doubles the rest go on the stack.
    """

    def __init__(self, name, params, body, ret, ret_ty):
        self.name = name
        self.params = params  # [(ty, name)]
        self.body = body
        self.ret = ret
        self.ret_ty = ret_ty

    def render(self):
        args = ", ".join(f"{t} {n}" for t, n in self.params)
        lines = [f"{self.ret_ty} {self.name}({args}) {{"]
        lines += [s.render("    ") for s in self.body]
        lines.append(f"    return {self.ret.render()};")
        lines.append("}")
        return "\n".join(lines)

    def call(self, argv):
        env = dict(zip([n for _, n in self.params], argv))
        for s in self.body:
            s.exec(env)
        v = self.ret.eval(env)
        return ckf(v) if self.ret_ty == "double" else ck(v)


# ── Generation ───────────────────────────────────────────────────────────────

ARITH = ["+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"]
CMP = ["<", ">", "<=", ">=", "==", "!="]


class Gen:
    def __init__(self, rng, shape):
        self.rng = rng
        self.shape = shape
        self.funcs = []

    # ── int expressions ──
    def expr(self, names, depth, arrays=()):
        r = self.rng
        if depth <= 0 or (names and r.random() < 0.3):
            if names and r.random() < 0.7:
                return Var(r.choice(names))
            return Const(r.randint(-100, 100))

        pick = r.random()
        if arrays and pick < 0.15:
            return Index(r.choice(arrays), self.expr(names, depth - 1))

        op = r.choice(ARITH)
        if op in ("<<", ">>"):
            return Bin(op, Bin("&", self.expr(names, depth - 1), Const(0xFF)),
                       Const(r.randint(0, 4)))
        if op in ("/", "%"):
            return Bin(op, self.expr(names, depth - 1),
                       Const(r.choice([2, 3, 4, 5, 7, 8, 11, 16, -3, -6])))
        if op in ("&", "|", "^"):
            return Bin(op, Bin("&", self.expr(names, depth - 1), Const(0x3FF)),
                       Const(r.randint(0, 0x3FF)))
        return Bin(op, self.expr(names, depth - 1), self.expr(names, depth - 1))

    # ── double expressions ──
    def fexpr(self, fnames, inames, depth):
        r = self.rng
        if depth <= 0 or (fnames and r.random() < 0.35):
            if fnames and r.random() < 0.75:
                return Var(r.choice(fnames))
            return FConst(r.randint(-40, 40))
        if inames and r.random() < 0.2:
            # int -> double conversion, the cvtsi2sd path
            return IntToF(Bin("&", self.expr(inames, 1), Const(0x3F)))
        return FBin(r.choice(["+", "-", "*"]),
                    self.fexpr(fnames, inames, depth - 1),
                    self.fexpr(fnames, inames, depth - 1))

    def cond(self, names, depth):
        return Bin(self.rng.choice(CMP), self.expr(names, depth), self.expr(names, depth))

    def stmts(self, names, arrays, n, depth, loop_ok=True, assignable=None):
        """`names` is readable; `assignable` is writable, and defaults to it.

        The two differ inside a loop body: the counter is in scope to read but
        must never be an assignment target. `i24 = p2` with `p2 == 0` resets the
        counter every iteration, so the loop never terminates -- and a program
        that does not terminate is one no oracle can check, which cost a session
        chasing two "miscompiles" that hung under `cc` too (seeds 14 and 29).
        Termination has to be unreachable-by-construction here, the same way
        UB-freedom is.
        """
        r = self.rng
        if assignable is None:
            assignable = names
        out = []
        for _ in range(n):
            pick = r.random()
            if pick < 0.5 or not assignable:
                out.append(Assign(r.choice(assignable), self.expr(names, depth, arrays)))
            elif pick < 0.62 and arrays:
                out.append(ArrAssign(r.choice(arrays), self.expr(names, 1),
                                     self.expr(names, depth, arrays)))
            elif pick < 0.8:
                out.append(If(self.cond(names, depth - 1),
                              self.stmts(names, arrays, r.randint(1, 2), depth - 1,
                                         loop_ok, assignable),
                              self.stmts(names, arrays, r.randint(1, 2), depth - 1,
                                         loop_ok, assignable)))
            elif loop_ok:
                var = f"i{r.randint(0, 99)}"
                body = self.stmts(names + [var], arrays, r.randint(1, 3), depth - 1,
                                  False, assignable)
                out.append(For(var, r.randint(1, 6), body))
            else:
                out.append(Assign(r.choice(assignable), self.expr(names, depth, arrays)))
        return out

    def wide_function(self, idx, scale=1.0):
        """7-12 parameters, mixed int/double: past the argument registers."""
        r = self.rng
        n = r.randint(7, 12) if scale >= 1.0 else r.randint(7, 8)
        params, inames, fnames = [], [], []
        for i in range(n):
            if self.shape == "args" or r.random() < 0.5:
                params.append(("int", f"p{i}"))
                inames.append(f"p{i}")
            else:
                params.append(("double", f"p{i}"))
                fnames.append(f"p{i}")
        if not inames:
            params[0] = ("int", "p0")
            inames.append("p0")

        nstmt = r.randint(1, 3) if scale >= 1.0 else 1
        body = self.stmts(inames, (), nstmt, 2 if scale >= 1.0 else 1, loop_ok=scale >= 1.0)
        if fnames and r.random() < 0.6:
            return Func(f"f{idx}", params, body, self.fexpr(fnames, inames, 2), "double")
        return Func(f"f{idx}", params, body, self.expr(inames, 2), "int")

    def call_of(self, fn, inames, fnames):
        args = []
        for ty, _ in fn.params:
            if ty == "int":
                args.append(self.expr(inames, 1))
            else:
                args.append(self.fexpr(fnames, inames, 1) if fnames else FConst(1))
        return Call(fn, args)


def build(seed, shape, n_int, n_dbl, scale=1.0):
    rng = random.Random(seed)
    g = Gen(rng, shape)
    for i in range(rng.randint(2, 3) if scale >= 1.0 else 1):
        g.funcs.append(g.wide_function(i, scale))

    inames = [f"v{i}" for i in range(n_int)]
    fnames = [f"d{i}" for i in range(n_dbl)]
    arrays = ["arr"]

    decls = [Decl("int", n, Const(rng.randint(-50, 50))) for n in inames]
    decls += [Decl("double", n, FConst(rng.randint(-30, 30))) for n in fnames]

    body = []
    # Pressure region: every value is defined before the calls and consumed
    # after them, so all of them are live across each call. All XMM registers
    # are caller-saved, so each live double must be routed through a slot.
    rounds = max(1, int(round(rng.randint(2, 4) * scale)))
    depth = 3 if scale >= 1.0 else 2
    for _ in range(rounds):
        body += g.stmts(inames, arrays, max(1, int(round(rng.randint(1, 3) * scale))), depth)
        for _ in range(rng.randint(1, 2)):
            fn = rng.choice(g.funcs)
            call = g.call_of(fn, inames, fnames)
            if fn.ret_ty == "double":
                body.append(Assign(rng.choice(fnames), call))
            else:
                body.append(Assign(rng.choice(inames), call))
        for n in fnames:
            body.append(Assign(n, FBin("+", Var(n), g.fexpr(fnames, inames, 1))))

    isum = Var(inames[0])
    for n in inames[1:]:
        isum = Bin("+", isum, Var(n))
    for i in range(ARRAY_SIZE):
        isum = Bin("+", isum, Index("arr", Const(i)))
    fsum = Var(fnames[0])
    for n in fnames[1:]:
        fsum = FBin("+", fsum, Var(n))

    env = {"arr": [i * 3 - 7 for i in range(ARRAY_SIZE)]}
    try:
        for d in decls:
            d.exec(env)
        for s in body:
            s.exec(env)
        i_result = isum.eval(env)
        f_result = fsum.eval(env)
    except (Reject, RecursionError, KeyError):
        return None
    if f_result != int(f_result):
        return None

    lines = [
        "// EXIT: 0",
        f"// OUTPUT: {i_result}",
        "// GENERATED by tests/fuzz/gen_c.py -- do not edit by hand.",
        "//",
        "// Free of undefined behavior by construction: pure expressions, no",
        "// unsequenced side effects, nonzero constant divisors, in-range shift",
        "// counts, indices masked into bounds, constant trip counts.",
        "// Doubles hold integral values and use only + - *, so every result is",
        "// exact and no reference compiler can legally disagree.",
        "",
        "extern int printf(char* fmt, ...);",
        "",
    ]
    lines += [f.render() + "\n" for f in g.funcs]
    lines.append("int main() {")
    lines.append(f"    int arr[{ARRAY_SIZE}];")
    for i in range(ARRAY_SIZE):
        lines.append(f"    arr[{i}] = {i * 3 - 7};")
    lines += [d.render("    ") for d in decls]
    lines += [s.render("    ") for s in body]
    # The double result is checked by comparison rather than printed: it is
    # exact, and this avoids depending on printf's float formatting.
    lines.append(f"    if ({fsum.render()} != {float(f_result):.1f}) {{ return 3; }}")
    lines.append(f'    printf("%d\\n", {isum.render()});')
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shape", default="mixed", choices=["mixed", "args", "pressure"])
    ap.add_argument(
        "--size",
        default="normal",
        choices=["small", "normal"],
        help="small trims statement count, nesting depth and live values, for "
        "reducing a failure to something that fits in a lit test",
    )
    ap.add_argument("--out")
    args = ap.parse_args()

    n_int, n_dbl = (10, 6)
    if args.shape == "pressure":
        n_int, n_dbl = (16, 12)  # more live values than there are registers
    scale = 1.0
    if args.size == "small":
        n_int, n_dbl = (4, 3)
        scale = 0.34

    # A rejected program means simulation caught a hazard; move to the next
    # seed rather than emitting anything questionable.
    src = None
    for attempt in range(400):
        src = build(args.seed * 1000 + attempt, args.shape, n_int, n_dbl, scale)
        if src is not None:
            break
    if src is None:
        print(f"gen_c.py: no valid program for seed {args.seed}", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w") as f:
            f.write(src)
    else:
        sys.stdout.write(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
