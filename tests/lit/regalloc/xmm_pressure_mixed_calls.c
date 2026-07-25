// KNOWN FAILING -- regression target for a real bug. Do not "fix" by
// weakening the test.
//
// Compilation aborts in the register allocator:
//
//   phase 'regalloc': global regalloc: register pressure overshoot for
//   function 'main' (gpr_overshoot=0, xmm_overshoot=1). The split pass
//   should have resolved all register pressure before phase 5.
//
// The splitter is supposed to have driven pressure below the register budget
// before the allocator runs, so reaching phase 5 over budget means the split
// pass missed a live range. Only 16 XMM registers exist and nothing here is
// remotely close to that, so this is a splitter defect, not real pressure.
//
// Minimal: it needs all three XMM-producing shapes at once. Any two of them
// compile fine:
//   - a call returning double whose argument is an int  (from_param)
//   - a call returning double taking mixed int/double   (mixed)
//   - an in-line char -> double conversion              (d = c)
//
// Found while writing tests/lit/float/int_to_double_conversions.c.
//
// OUTPUT: a: 10.000000
// OUTPUT: c: 15.700000
// OUTPUT: d: -5.000000
// EXIT: 0

extern int printf(char* fmt, double x);

double from_param(int n) {
    return n;
}

double mixed(int n, double x) {
    return n + x;
}

int main() {
    printf("a: %f\n", from_param(10));
    printf("c: %f\n", mixed(10, 5.7));

    char c;
    c = -5;
    double d;
    d = c;
    printf("d: %f\n", d);
    return 0;
}
