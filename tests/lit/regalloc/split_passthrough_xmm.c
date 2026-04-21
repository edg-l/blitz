// Test: XMM value live across a call in a pass-through block.
//
// An XMM (double) value defined before the call block passes THROUGH the
// call block (block B) without being used there, and is used AFTER the call
// in the merge block (block C). Since all XMM registers are caller-saved in
// SysV ABI, the value must be preserved across the call even though block B
// does not use it.
//
// This test guards against the "block-local liveness only" bug: a splitter
// that seeds its backward liveness scan from scratch (not from
// GlobalLiveness::live_in) would fail to see the XMM value as live in block
// B and would not insert the required spill/reload pair.
//
// With BLITZ_SPLIT=1: the splitter detects XMM pressure in block B
// (base is live-in from block A, even though block B doesn't use it),
// inserts a slot spill before the call and reloads after, producing correct
// output.
//
// Without BLITZ_SPLIT=1: the global allocator handles this via its existing
// cross-block spill logic.
//
// OUTPUT: 3.500000
// EXIT: 0

extern int printf(char* fmt, double x);

__attribute__((noinline))
double identity(double x) {
    return x;
}

int main() {
    double base = 2.5;
    double mid_result = identity(1.0);
    double result = base + mid_result;
    printf("%f\n", result);
    return 0;
}
