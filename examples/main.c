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

int main(void) {
    printf("add(3, 4)    = %ld\n", add(3, 4));
    printf("max(10, 20)  = %ld\n", max(10, 20));
    printf("max(-5, 3)   = %ld\n", max(-5, 3));
    printf("sum_to(100)  = %ld\n", sum_to(100));

    // Verify
    int ok = 1;
    ok &= (add(3, 4) == 7);
    ok &= (max(10, 20) == 20);
    ok &= (max(-5, 3) == 3);
    ok &= (sum_to(100) == 5050);

    if (ok) {
        printf("\nAll checks passed!\n");
        return 0;
    } else {
        printf("\nSome checks FAILED!\n");
        return 1;
    }
}
