// Example C main to link with Blitz-compiled output.o
//
// Build:
//   cargo run --example basic
//   cc main.c output.o -o test
//   ./test

#include <stdint.h>
#include <stdio.h>

int64_t add(int64_t a, int64_t b);
int64_t max(int64_t a, int64_t b);
int64_t sum_to(int64_t n);
int64_t optimized(int64_t x);
int64_t array_idx(int64_t base, int64_t i);

int main(void) {
    printf("add(3, 4)       = %ld\n", add(3, 4));
    printf("max(10, 20)     = %ld\n", max(10, 20));
    printf("max(-5, 3)      = %ld\n", max(-5, 3));
    printf("sum_to(100)     = %ld\n", sum_to(100));
    printf("optimized(42)   = %ld\n", optimized(42));
    printf("array_idx(100, 3) = %ld\n", array_idx(100, 3));

    // Verify
    int ok = 1;
    ok &= (add(3, 4) == 7);
    ok &= (max(10, 20) == 20);
    ok &= (max(-5, 3) == 3);
    ok &= (sum_to(100) == 5050);
    ok &= (optimized(42) == -1);        // (42/1) | -1 = -1
    ok &= (array_idx(100, 3) == 140);   // 100 + 3*8 + 16 = 140

    if (ok) {
        printf("\nAll checks passed!\n");
        return 0;
    } else {
        printf("\nSome checks FAILED!\n");
        return 1;
    }
}
