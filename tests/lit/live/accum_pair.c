// A reduction split across four independent accumulators, beside the serial
// one it is supposed to beat.
//
// The four partials have no dependence on each other, so they are the shape a
// machine with four ALU ports is meant to overlap; the serial chain right after
// them is the same trip count with a multiply on the critical path. Both keep
// their accumulator live across the whole loop, so this is five values that
// must stay in registers over a body that also loads four array elements.

// OUTPUT: 653568
// OUTPUT: 8448
// OUTPUT: 518144
// OUTPUT: 250880
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int vals[512];
    int chk0 = 0;
    int chk1 = 0;
    int chk2 = 0;
    int chk3 = 0;
    int reps = 103 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 61) & 255;

        for (int i = 0; i < 512; i = i + 1) {
            vals[i] = (i * 331 + seed) & 8191;
        }

        int a0 = 0;
        int a1 = 0;
        int a2 = 0;
        int a3 = 0;
        for (int pass = 0; pass < 4; pass = pass + 1) {
            for (int i = 0; i < 512; i = i + 4) {
                a0 = (a0 + vals[i] + pass) & 1048575;
                a1 = (a1 + (vals[i + 1] ^ pass)) & 1048575;
                a2 = (a2 + vals[i + 2] * 3) & 1048575;
                a3 = (a3 ^ (vals[i + 3] << 1)) & 1048575;
            }
        }

        int serial = 0;
        for (int pass = 0; pass < 4; pass = pass + 1) {
            for (int i = 0; i < 512; i = i + 1) {
                serial = (serial * 3 + vals[i]) & 1048575;
            }
        }

        chk0 = (chk0 + (a0)) & 1048575;
        chk1 = (chk1 + (a1)) & 1048575;
        chk2 = (chk2 + ((a2 + a3) & 1048575)) & 1048575;
        chk3 = (chk3 + (serial)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    printf("%d\n", chk2);
    printf("%d\n", chk3);
    return 0;
}
