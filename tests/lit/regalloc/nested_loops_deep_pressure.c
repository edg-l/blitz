// Test: doubly-nested loop with a call inside the inner body.
//
// This stresses the register allocator's handling of values that must remain
// live across many block boundaries (outer loop counter i, inner loop counter
// j, running sum) while a call inside the inner body clobbers all caller-saved
// registers every iteration.
//
// CURRENT STATUS: FAILING under the per-block allocator — runtime segfault.
// Any doubly-nested loop where the inner body contains a `noinline` call
// crashes at execution. Single-loop-with-call works; adding a nesting level
// breaks it. Root cause is likely a block-param or phi-copy miscompilation in
// the per-block cross-block spill/reload path. The global allocator (Phase 6)
// is expected to resolve this by handling cross-block liveness in one pass
// instead of patching it together via `split.rs`.
//
// Do NOT work around this failure by removing the call or the nesting — the
// failing test is a regression target for the global allocator work.
//
// EXIT: 0

__attribute__((noinline))
int step(int x) {
    return x + 1;
}

int main() {
    int sum = 0;
    int i = 0;
    while (i < 5) {
        int j = 0;
        while (j < 5) {
            sum = step(sum) + i + j - 1;
            j = j + 1;
        }
        i = i + 1;
    }
    // Each inner iter: sum = sum + 1 + i + j - 1 = sum + i + j
    // => sum_{i=0}^{4} sum_{j=0}^{4} (i+j) = 100
    if (sum != 100) { return 1; }
    return 0;
}
