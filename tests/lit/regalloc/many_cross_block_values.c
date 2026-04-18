// Test: multiple values live across block boundaries in a loop.
//
// This test exercises the per-block allocator's cross-block spill/reload
// machinery. Values a, b, c, d are computed before the loop and used inside
// the loop body after a conditional branch. Under the current per-block
// allocator each cross-block value travels through a stack slot at every
// block boundary; the global allocator (Phase 6) should keep them in
// registers across boundaries when pressure allows.
//
// POST-CUTOVER: after Phase 6 lands, CHECK-COUNT-N for mov [rsp...] should
// decrease significantly. The current per-block path emits at least 4
// spill/reload sequences for these 4 values. The global allocator target
// is 0 (all fit in the 15-register budget).
//
// OUTPUT: 65
// EXIT: 0

extern int printf(char* fmt, int x);

__attribute__((noinline))
int compute(int n) {
    int a = n + 1;
    int b = n + 2;
    int c = n + 3;
    int d = n + 4;
    int sum = 0;
    int i = 0;
    while (i < 5) {
        if (i > 0) {
            sum = sum + a + b;
        }
        sum = sum + c + d;
        i = i + 1;
    }
    return sum;
}

int main() {
    int result = compute(1);
    printf("%d\n", result);
    return 0;
}
