// Test (shadow): XMM value live across a call in a non-loop context.
//
// A simpler variant of xmm_loop_crossing.c without the loop, testing that a
// single XMM value is correctly preserved across a call to a noinline helper.
//
// This test passes both with and without BLITZ_SPLIT=1. It guards against
// regressions in XMM-across-call handling after the splitter is integrated.
//
// OUTPUT: 3.750000
// EXIT: 0

extern int printf(char* fmt, double x);

__attribute__((noinline))
double scale(double x) {
    return x * 2.0;
}

int main() {
    double base = 1.25;
    double result = scale(base) + base;
    printf("%f\n", result);
    return 0;
}
