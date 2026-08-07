// A histogram: a load, an add and a store all through an index the loop just
// computed, then a gather that reads the table back through a second one.
//
// The scatter is the point. Every iteration reads and writes `hist` at a
// runtime index, so no store can be forwarded to the next load and no store can
// be proven dead -- an alias question the backend has to answer conservatively
// on every pass. The gather afterwards is the same address shape without the
// write, which is what the folded addressing mode should be reducing to one
// operand.

// OUTPUT: 303104
// OUTPUT: 327680
// OUTPUT: 74
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int vals[512];
    int hist[64];
    int chk0 = 0;
    int chk1 = 0;
    int chk2 = 0;
    int reps = 74 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 149) & 511;

        for (int i = 0; i < 512; i = i + 1) {
            vals[i] = (i * 773 + seed) & 65535;
        }
        for (int i = 0; i < 64; i = i + 1) {
            hist[i] = 0;
        }

        for (int pass = 0; pass < 8; pass = pass + 1) {
            for (int i = 0; i < 512; i = i + 1) {
                int b = (vals[i] >> 3) & 63;
                if (b > 40) {
                    b = b - 40;
                }
                hist[b] = hist[b] + 1;
            }
        }

        int total = 0;
        for (int i = 0; i < 64; i = i + 1) {
            total = total + hist[i];
        }

        int gathered = 0;
        for (int i = 0; i < 512; i = i + 1) {
            gathered = (gathered + hist[vals[i] & 63]) & 1048575;
        }

        int peak = 0;
        int best = 0;
        for (int i = 0; i < 64; i = i + 1) {
            if (hist[i] > best) {
                best = hist[i];
                peak = i;
            }
        }

        chk0 = (chk0 + (total)) & 1048575;
        chk1 = (chk1 + (gathered)) & 1048575;
        chk2 = (chk2 + (peak)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    printf("%d\n", chk2);
    return 0;
}
