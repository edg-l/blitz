// Regression: signed int -> double conversions.
//
// Two bugs lived here. Both were found by the -O0/-O1 differential harness
// (tests/lit/run_diff.sh) plus a gcc cross-check.
//
// 1. An integer parameter feeding cvtsi2sd was assigned an XMM register:
//    build_vreg_classes_from_insts inferred operand class from is_fp_op(),
//    which describes the *result*. The encoder then wrote the XMM index into a
//    GPR field, so `double g(int n) { return n; }` compiled to
//    `cvtsi2sd xmm0, rsp` and returned a stack address.
//
// 2. cvtsi2sd was always emitted with REX.W, reading 64 bits of a register
//    holding a 32-bit value. SysV leaves those high bits undefined, so
//    negative ints came back as large positives: g(-5) gave 4294967291.0.
//
// OUTPUT: from_param: 10.000000
// OUTPUT: negative: -5.000000
// OUTPUT: from_char: -5.000000
// OUTPUT: from_long: 5000000000.000000
// EXIT: 0

extern int printf(char* fmt, double x);

double from_param(int n) {
    return n;
}

double from_long(long n) {
    return n;
}

int main() {
    printf("from_param: %f\n", from_param(10));
    printf("negative: %f\n", from_param(-5));

    char c;
    c = -5;
    double d;
    d = c;
    printf("from_char: %f\n", d);
    printf("from_long: %f\n", from_long(5000000000));
    return 0;
}
