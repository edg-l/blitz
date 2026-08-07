// Regression test: both the quotient and the remainder of one division are
// consumed, so the pair has to leave RAX and RDX together.
//
// `idiv` writes both at once and the projections copy them out, which makes the
// extraction a parallel copy, not two independent moves. Emitted as two moves the
// first can destroy what the second reads: with the quotient's destination
// allocated RDX, `mov edx,eax` overwrote the remainder, and `a / b + a % b` came
// out as twice the quotient.
//
// The two are emitted as one sequentialized copy, which breaks the RAX/RDX cycle
// through the scratch register.
//
// `opaque` keeps the operands from being folded, so a real idiv is emitted.

// FLAGS: -O0
// EXIT: 0
// OUTPUT: 7
// OUTPUT: 4
// OUTPUT: -7
// OUTPUT: 5

extern int printf(char* fmt, ...);

int opaque(int x) { return x; }

int divmod(int a, int b) { return (a / b) + (a % b); }

// The reverse order, in case one order happens to allocate its way around the
// collision.
int moddiv(int a, int b) { return (a % b) + (a / b); }

int main() {
    // 17/3 == 5, 17%3 == 2.
    printf("%d\n", divmod(opaque(17), opaque(3)));
    // 10/3 == 3, 10%3 == 1.
    printf("%d\n", divmod(opaque(10), opaque(3)));
    // -17/3 == -5, -17%3 == -2.
    printf("%d\n", divmod(opaque(-17), opaque(3)));
    // 17%4 == 1, 17/4 == 4.
    printf("%d\n", moddiv(opaque(17), opaque(4)));
    return 0;
}
