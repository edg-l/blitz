// A search loop that leaves early, run 200 times against runtime data.
//
// The inner loop has two exits and a trip count the compiler cannot know, so
// the loop-closing block takes a parameter from both. That is the shape where
// LICM has to prove a hoist is safe on a path that may not execute, and where
// the block-parameter machinery earns its keep.

// OUTPUT: 1100
// OUTPUT: 139865
// OUTPUT: 422565
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int data[256];
    int chk0 = 0;
    int chk1 = 0;
    int chk2 = 0;
    int reps = 11 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 13) & 63;

        for (int i = 0; i < 256; i = i + 1) {
            data[i] = (i * 29 + seed) & 511;
        }

        int found = 0;
        int total = 0;
        int scanned = 0;
        for (int t = 0; t < 200; t = t + 1) {
            int target = (t * 7 + seed) & 511;
            int idx = 0 - 1;
            for (int i = 0; i < 256; i = i + 1) {
                scanned = (scanned + 1) & 65535;
                if (data[i] == target) {
                    idx = i;
                    break;
                }
            }
            if (idx >= 0) {
                found = found + 1;
                total = (total + idx) & 65535;
            }
        }

        chk0 = (chk0 + (found)) & 1048575;
        chk1 = (chk1 + (total)) & 1048575;
        chk2 = (chk2 + (scanned)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    printf("%d\n", chk2);
    return 0;
}
