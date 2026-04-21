// Test: XMM block param with multiple predecessors, one of which makes a call.
//
// A merge block receives an XMM value from two different predecessors. One
// predecessor path makes a call (which clobbers all XMM registers in SysV ABI),
// so the block param at the merge point must be slot-spilled. The other
// predecessor path has no call and provides its value via a direct slot store
// at the terminator.
//
// This guards against the multi-predecessor case: when only some predecessors
// cross a call boundary, the SlotSpillBlockParam strategy must still emit a
// slot store on ALL predecessors (including call-free ones) so the merge block
// can uniformly reload from the slot.
//
// Requires BLITZ_SPLIT=1.
//
// OUTPUT: 7.500000
// EXIT: 0

extern int printf(char* fmt, double x);

__attribute__((noinline))
double transform(double x) {
    return x * 2.0;
}

int main() {
    int flag = 1;
    double val;
    if (flag) {
        // This path calls transform, clobbering XMM registers.
        val = transform(2.5);
    } else {
        // This path has no call.
        val = 5.0;
    }
    // val is 5.0 from the call path (2.5 * 2.0 = 5.0)
    // Multiply by 1.5 => 7.5
    double result = val * 1.5;
    printf("%f\n", result);
    return 0;
}
