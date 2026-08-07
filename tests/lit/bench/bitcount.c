// Population count over 1024 words, twice: bit-by-bit and by Kernighan's
// clear-lowest-set-bit loop.
//
// Two ways to compute one answer, so the pair is a standing check that the
// arithmetic rules did not change what the program means. Kernighan's inner
// loop has a data-dependent trip count; the naive one is a fixed 32.

// OUTPUT: 16321
// OUTPUT: 0
// EXIT: 0


extern int printf(char* fmt, ...);

int popcount_naive(unsigned int v) {
    int n = 0;
    for (int i = 0; i < 32; i = i + 1) {
        n = n + (int)((v >> i) & 1);
    }
    return n;
}

int popcount_kernighan(unsigned int v) {
    int n = 0;
    while (v != 0) {
        v = v & (v - 1);
        n = n + 1;
    }
    return n;
}

int main() {
    unsigned int words[1024];
    unsigned int seed = 88172645;
    for (int i = 0; i < 1024; i = i + 1) {
        seed = seed ^ (seed << 13);
        seed = seed ^ (seed >> 17);
        seed = seed ^ (seed << 5);
        words[i] = seed;
    }

    int total_naive = 0;
    int total_kern = 0;
    for (int i = 0; i < 1024; i = i + 1) {
        total_naive = total_naive + popcount_naive(words[i]);
        total_kern = total_kern + popcount_kernighan(words[i]);
    }

    printf("%d\n", total_naive);
    printf("%d\n", total_naive - total_kern);
    return 0;
}
