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

// OUTPUT: 635648
// OUTPUT: 635648
// OUTPUT: 9024
// OUTPUT: 324480
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    unsigned char src[256];
    unsigned char dst[256];
    unsigned char tmp[256];
    int chk0 = 0;
    int chk1 = 0;
    int chk2 = 0;
    int chk3 = 0;
    int reps = 1690 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 53) & 255;

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

        chk0 = (chk0 + (sum_dst)) & 1048575;
        chk1 = (chk1 + (sum_tmp)) & 1048575;
        chk2 = (chk2 + (filled)) & 1048575;
        chk3 = (chk3 + (first_diff)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    printf("%d\n", chk2);
    printf("%d\n", chk3);
    return 0;
}
