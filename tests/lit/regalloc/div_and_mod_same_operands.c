// Regression test: taking both the quotient and the remainder of one division,
// with a call between the two extractions.
//
// `idiv` writes both results at once, the quotient in RAX and the remainder in
// RDX, and nothing in the IR says those registers hold anything: the projection
// that copies a result into a VReg is the only reader, and no interference edge
// records the dependence. Two passes each took the schedule at face value.
//
// Barrier grouping pulls a definition toward its consumer to shorten its live
// range, which moved the quotient's projection past the `printf` that consumes
// the remainder -- and a call destroys RAX and RDX, so the projection copied out
// whatever `printf` had returned. Lowering then compounded it: it runs on each
// run of pure ops between barriers and asked whether the projected VReg came from
// a division by looking only inside that run, so with the division in an earlier
// run the answer was no, and the projection was lowered as an ordinary register
// copy from the pair VReg -- a register nothing writes.
//
// A division's projections belong in the division's own group, and the questions
// lowering asks about a projection have to be answered over the whole block.
//
// `opaque` keeps the operands out of reach of constant folding; folded away, the
// division never reaches the allocator at all.

// FLAGS: -O0
// EXIT: 0
// OUTPUT: 0
// OUTPUT: 6
// OUTPUT: 0
// OUTPUT: 6
// OUTPUT: 2
// OUTPUT: -4
// OUTPUT: 3
// OUTPUT: 3

extern int printf(char* fmt, int x);

int opaque(int x) { return x; }

int main() {
    int a = opaque(18);
    int b = opaque(-18);
    int d = opaque(5);

    // Remainder first, then quotient: the quotient's extraction is what gets
    // separated from the division.
    printf("%d\n", (a % 3));
    printf("%d\n", (a / 3));

    // A negative dividend, so a wrong sign shows up as a wrong value.
    printf("%d\n", (b % 3));
    printf("%d\n", (b / -3));

    // A negative divisor, where reading the constant as unsigned would give
    // a % -4 == a and a / -4 == 0.
    printf("%d\n", (a % -4));
    printf("%d\n", (a / -4));

    // A divisor the compiler cannot see, so no strength reduction applies.
    printf("%d\n", (a % d));
    printf("%d\n", (a / d));

    return 0;
}
