// Two floating-point reductions over runtime data: a dot product and a
// scaled accumulation.
//
// The values stay integral and use only + - *, so every result is exact and no
// reference compiler can legally reassociate its way to a different answer.
// That is what makes the output comparable across compilers at all.

// OUTPUT: 1006976
// OUTPUT: 824320
// EXIT: 0

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    double xs[256];
    double ys[256];
    int chk0 = 0;
    int chk1 = 0;
    int reps = 219 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 3) & 7;

        for (int i = 0; i < 256; i = i + 1) {
            xs[i] = (double)((i + seed) & 63);
            ys[i] = (double)((i * 2 + 1) & 63);
        }

        double dot = 0.0;
        for (int i = 0; i < 256; i = i + 1) {
            dot = dot + xs[i] * ys[i];
        }

        double scaled = 0.0;
        for (int pass = 0; pass < 4; pass = pass + 1) {
            for (int i = 0; i < 256; i = i + 1) {
                scaled = scaled + (xs[i] - ys[i]) * 2.0;
            }
        }

        chk0 = (chk0 + ((int)dot)) & 1048575;
        chk1 = (chk1 + ((int)scaled)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
