// Test: XMM value live across a call inside a loop body.
//
// An XMM (double) value defined before the loop is used inside the body after
// a call. Since all XMM registers are caller-saved in SysV ABI, the value must
// travel through a spill slot across the call on every iteration. This is the
// R4 requirement (XMM-across-call forced spill).
//
// CURRENT STATUS: FAILING under the per-block allocator. Reproduces
// "XMM chromatic=17 (avail=16)" in block 2 — the per-block scope cannot reduce
// XMM pressure below 17 within MAX_SPILL_ROUNDS because all XMM registers are
// clobbered by the call and the per-block spill splitter cannot model the
// cross-iteration liveness correctly. The global allocator (Phase 6) is
// expected to handle this via function-scope pressure-based spill selection.
//
// Do NOT work around this failure by removing the loop — the failing test is a
// regression target for the global allocator work.
//
// OUTPUT: 11.250000
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
    // acc = 5 * (1.25 * 2.0) + 1.25 = 11.25
    acc = acc + base;
    printf("%f\n", acc);
    return 0;
}
