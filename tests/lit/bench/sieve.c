// Sieve of Eratosthenes over a 10000-element flag array.
//
// A counted inner loop whose stride is the outer loop's induction variable, so
// the address `flags[j]` is `base + j*4` with `j` advancing by `p` -- the shape
// loop strength reduction exists for.

// OUTPUT: 1229
// OUTPUT: 9973
// EXIT: 0


extern int printf(char* fmt, ...);

int main() {
    int flags[10000];
    for (int i = 0; i < 10000; i = i + 1) {
        flags[i] = 1;
    }
    flags[0] = 0;
    flags[1] = 0;

    for (int p = 2; p * p < 10000; p = p + 1) {
        if (flags[p]) {
            for (int j = p * p; j < 10000; j = j + p) {
                flags[j] = 0;
            }
        }
    }

    int count = 0;
    int last = 0;
    for (int i = 0; i < 10000; i = i + 1) {
        if (flags[i]) {
            count = count + 1;
            last = i;
        }
    }
    printf("%d\n", count);
    printf("%d\n", last);
    return 0;
}
