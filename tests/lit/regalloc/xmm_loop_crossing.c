// Test: XMM value live across a call inside a loop body.
//
// An XMM (double) value defined before the loop is used inside the body after
// a call. Since all XMM registers are caller-saved in SysV ABI, the value must
// travel through a spill slot across the call on every iteration. This is the
// R4 requirement (XMM-across-call forced spill).
//
// Requires BLITZ_SPLIT=1: the global allocator with Phase 6 block-param
// slot-spilling handles this correctly by routing the XMM block params through
// stack slots when they are live across calls.
//
// OUTPUT: 13.750000
// EXIT: 0

extern int printf(char* fmt, double x);

__attribute__((noinline))
double scale(double x) {
    return x * 2.0;
}

int main() {
    double base = 1.25;
    double acc = 0.0;
    int i = 0;
    while (i < 5) {
        acc = acc + scale(base);
        i = i + 1;
    }
    // acc = 5 * scale(1.25) + base = 5 * 2.5 + 1.25 = 13.75
    acc = acc + base;
    printf("%f\n", acc);
    return 0;
}
