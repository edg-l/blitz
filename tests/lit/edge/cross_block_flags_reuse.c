// EXIT: 2
// XFAIL: cross-block flags reuse bug -- EFLAGS from block0's x>5 comparison
// gets clobbered by block3's x>20 subtraction, but the backend reuses the
// stale proj1 value instead of recomputing the comparison.

int main() {
    int x = 10;
    if (x > 5) {
        if (x > 20) {
            return 1;
        }
        // This should be true (10 > 5), but the backend reuses stale flags
        // from the x > 20 comparison above.
        if (x > 5) {
            return 2;
        }
    }
    return 3;
}
