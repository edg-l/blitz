// A floating-point argument past XMM7 travels on the stack, and `push`
// addresses no XMM register.
//
// SysV gives the first eight floating-point arguments XMM0-XMM7 and puts the
// rest in the argument area, where the callee reads them with `movsd`. Both
// ends of that are a register class the integer path cannot name: the caller
// has to move the value into a GPR to push it, and the callee's load is `movsd`
// rather than a `mov` of some `OpSize`.
//
// `mix` is the same question with the two sequences interleaved: it runs out of
// GPRs and XMMs at different arguments, so a stack slot is claimed by an
// integer and the next by a double.
//
// RUN: %tinyc %s --emit-asm 2>&1
// CHECK-LABEL: # main
// CHECK: movq   {{r[a-z0-9]+}},xmm{{[0-9]+}}
// CHECK: push
// OUTPUT: 1987654321
// OUTPUT: 87654321
// OUTPUT: 616173
// EXIT: 0
extern int printf(char* fmt, int x);

__attribute__((noinline))
double ten_doubles(double a, double b, double c, double d, double e, double f,
                   double g, double h, double i, double j) {
    return a * 1.0 + b * 10.0 + c * 100.0 + d * 1000.0 + e * 10000.0
         + f * 100000.0 + g * 1000000.0 + h * 10000000.0
         + i * 100000000.0 + j * 1000000000.0;
}

__attribute__((noinline))
int eight_ints(int a, int b, int c, int d, int e, int f, int g, int h) {
    return a + b * 10 + c * 100 + d * 1000
         + e * 10000 + f * 100000 + g * 1000000 + h * 10000000;
}

__attribute__((noinline))
double mix(int a, double b, int c, double d, int e, double f, int g, double h,
           int i, double j, int k, double l) {
    return (double)(a + c * 10 + e * 100 + g * 1000 + i * 10000 + k * 100000)
         + b + d * 10.0 + f * 100.0 + h * 1000.0 + j * 10000.0 + l * 100000.0;
}

int main() {
    printf("%d\n", (int)ten_doubles(1.0, 2.0, 3.0, 4.0, 5.0,
                                    6.0, 7.0, 8.0, 9.0, 1.0));
    printf("%d\n", eight_ints(1, 2, 3, 4, 5, 6, 7, 8));
    printf("%d\n", (int)mix(1, 2.0, 3, 4.0, 5, 6.0, 7, 8.0, 9, 1.0, 2, 3.0));
    return 0;
}
