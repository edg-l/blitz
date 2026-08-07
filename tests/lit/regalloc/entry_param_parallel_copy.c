// The moves that place incoming arguments in the registers the allocator chose
// for them are one parallel copy, not a sequence.
//
// A parameter in a caller-saved register is not pre-colored when its block
// contains a call, so it gets an ordinary register and an entry move instead.
// Emitting those moves in list order lets one destroy an argument a later one
// still has to read: with the arguments in RDI..R9 and `a` assigned RCX,
// `mov rcx, rdi` overwrites the fourth argument before `mov rdi, rcx` reads it,
// and `d` silently takes `a`'s value. Six arguments summed came out
// `a + b + c + a + c + f` = 16 rather than 21, and the same shape at `-O0` gave
// every parameter one scratch register and stored that one value into all six
// slots, for 6 * 6 = 36.
//
// Both are the same fact from two sides: every parameter is already in its
// argument register before the function's first instruction runs.

extern int printf(char* fmt, int x);

__attribute__((noinline))
int six(int a, int b, int c, int d, int e, int f) {
    printf("%d\n", a + b + c + d + e + f);
    return a - f;
}

__attribute__((noinline))
int five(int a, int b, int c, int d, int e) {
    printf("%d\n", a * 10000 + b * 1000 + c * 100 + d * 10 + e);
    return 0;
}

int main() {
    six(1, 2, 3, 4, 5, 6);
    five(1, 2, 3, 4, 5);
    return 0;
}

// OUTPUT: 21
// OUTPUT: 12345
// EXIT: 0
