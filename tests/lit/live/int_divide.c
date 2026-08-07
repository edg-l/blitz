// Integer division and remainder in a loop, by a divisor that changes every
// iteration and that no compiler can see.
//
// `idiv` writes both quotient and remainder and the ABI pins them to rax and
// rdx, so this is the kernel where precoloring, the div/mod pairing and the
// live range that has to survive a two-register clobber all show up at once.
// The divisor is masked and or-ed with 1, so it is never zero.

// OUTPUT: 157022
// OUTPUT: 251732
// EXIT: 0

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    int vals[128];
    int chk0 = 0;
    int chk1 = 0;
    int reps = 142 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 37) & 255;

        for (int i = 0; i < 128; i = i + 1) {
            vals[i] = (i * 977 + seed) & 65535;
        }

        int q = 0;
        int r = 0;
        for (int pass = 0; pass < 8; pass = pass + 1) {
            for (int i = 0; i < 128; i = i + 1) {
                int d = ((vals[i] & 31) | 1) + pass;
                q = (q + vals[i] / d) & 262143;
                r = (r + vals[i] % d) & 8191;
            }
        }

        chk0 = (chk0 + (q)) & 1048575;
        chk1 = (chk1 + (r)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
