// An odd number of stack arguments leaves RSP misaligned at the call.
//
// SysV requires RSP % 16 == 0 at a `call`, and `setup_call_args` emits one
// `push` per stack argument. Seven integer arguments put one on the stack, so
// one push runs and the callee starts with RSP off by eight; its own call into
// libc then hands glibc a misaligned stack and the SSE path in `printf` faults.
// Even counts happen to survive, which is why nothing has caught this: the
// generator's `args` shape and every corpus program land on even counts.
//
// Segfaults at both levels, so the -O0-vs-O1 leg agrees and only the reference
// compiler sees it.
//
// OUTPUT: 21.000000
// OUTPUT: 28.000000
// OUTPUT: 36.000000
extern int printf(char* fmt, double x);

__attribute__((noinline))
double six(int a, int b, int c, int d, int e, int f) {
    printf("%f\n", (double)(a + b + c + d + e + f));
    return 0.0;
}

__attribute__((noinline))
double seven(int a, int b, int c, int d, int e, int f, int g) {
    printf("%f\n", (double)(a + b + c + d + e + f + g));
    return 0.0;
}

__attribute__((noinline))
double eight(int a, int b, int c, int d, int e, int f, int g, int h) {
    printf("%f\n", (double)(a + b + c + d + e + f + g + h));
    return 0.0;
}

int main() {
    six(1, 2, 3, 4, 5, 6);
    seven(1, 2, 3, 4, 5, 6, 7);
    eight(1, 2, 3, 4, 5, 6, 7, 8);
    return 0;
}
