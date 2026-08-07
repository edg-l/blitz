// A loop that crosses the register file on every iteration: int to double, a
// double multiply-add, and double back to int, with the data seeded at runtime.
//
// Every value here changes register class twice per iteration, so `cvtsi2sd`
// and `cvttsd2si` read the class opposite the one they write and the allocator
// has to keep an integer and a vector working set live at once. The arithmetic
// is exact in binary floating point (1.5 and 0.25, over small integers), so the
// answer does not depend on rounding.

// OUTPUT: 525776
// OUTPUT: 210432
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int chk0 = 0;
    int chk1 = 0;
    int reps = 1065 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 5) & 7;
        double acc = 0.0;
        int total = 0;

        for (int i = 0; i < 256; i = i + 1) {
            int v = ((i * 17 + seed) & 63) - 20;
            double d = (double)v;
            d = d * 1.5 + 0.25;
            int back = (int)d;
            total = total + back;
            acc = acc + d;
        }

        // Scaled by four so the quarters the truncation threw away are visible:
        // `total` and `acc` must not agree by accident.
        chk0 = (chk0 + (total)) & 1048575;
        chk1 = (chk1 + ((int)(acc * 4.0))) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
