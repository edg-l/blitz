// Test: high register pressure around a call site.
//
// Computes several independent sub-expressions, calls a helper function in
// the middle, then combines all results. This forces:
//   - Call-arg precoloring (first arg -> RDI)
//   - Caller-saved clobber interference (values computed before the call must
//     survive into callee-saved registers or be spilled)
//   - Cross-block liveness (values computed in the entry block must remain
//     accessible after the call returns)
//
// The 8 values a..h are all live across the helper call. Under the per-block
// allocator, caller-saved values are spilled before the call and reloaded
// after. The global allocator (Phase 6) should place them in callee-saved
// registers when possible.
//
// OUTPUT: 88

extern int printf(char* fmt, int x);

__attribute__((noinline))
int helper(int x) {
    return x * 2;
}

int main() {
    int a = 3;
    int b = 5;
    int c = 7;
    int d = 9;
    int e = 11;
    int f = 13;
    int g = 15;
    int h = 17;

    int mid = helper(a + b);

    int result = mid + c + d + e + f + g + h;
    printf("%d\n", result);
    return 0;
}
