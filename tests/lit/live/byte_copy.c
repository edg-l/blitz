// memcpy-shaped byte traffic: a forward copy, a backward copy over the same
// buffer, a fill, and a compare loop that stops at the first difference.
//
// Every iteration is a byte load feeding a byte store, so the loop body is two
// memory operands and an index increment and nothing else. That leaves the
// address arithmetic with nowhere to hide: a copy that recomputes `base + i`
// from scratch, or spills the destination pointer across the body, shows up as
// a multiple of the instruction count rather than a constant. The backward copy
// runs the index the other way, which strength reduction has to handle without
// turning the bound into an unsigned wrap. `argc` seeds the data so no
// reference compiler can evaluate the program and print the answer.

// OUTPUT: 32640
// OUTPUT: 32640
// OUTPUT: 28448
// OUTPUT: 192
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    unsigned char src[256];
    unsigned char dst[256];
    unsigned char tmp[256];
    int seed = (argc * 53) & 255;

    for (int i = 0; i < 256; i = i + 1) {
        src[i] = (unsigned char)(((i * 5) + seed) & 255);
        dst[i] = (unsigned char)0;
        tmp[i] = (unsigned char)0;
    }

    // Forward copy, the plain memcpy shape.
    for (int i = 0; i < 256; i = i + 1) {
        dst[i] = src[i];
    }

    // Backward copy of the same bytes into a third buffer.
    for (int i = 255; i >= 0; i = i - 1) {
        tmp[i] = dst[i];
    }

    int sum_dst = 0;
    int sum_tmp = 0;
    for (int i = 0; i < 256; i = i + 1) {
        sum_dst = sum_dst + (int)dst[i];
        sum_tmp = sum_tmp + (int)tmp[i];
    }

    // Fill the tail, then a compare loop that stops at the first difference.
    for (int i = 192; i < 256; i = i + 1) {
        tmp[i] = (unsigned char)(seed & 127);
    }

    int filled = 0;
    for (int i = 0; i < 256; i = i + 1) {
        filled = filled + (int)tmp[i];
    }

    int first_diff = 256;
    int i = 0;
    while (i < 256) {
        if (dst[i] != tmp[i]) {
            first_diff = i;
            i = 256;
        } else {
            i = i + 1;
        }
    }

    printf("%d\n", sum_dst);
    printf("%d\n", sum_tmp);
    printf("%d\n", filled);
    printf("%d\n", first_diff);
    return 0;
}
