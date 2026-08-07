// A loop whose branch outcome depends on data the compiler cannot know.
//
// Nothing here folds into straight-line code, so the block layout, the
// comparison and the conditional jump are all real. This is the shape where
// branch layout and `cmov`-versus-branch selection decide the answer.

// OUTPUT: 918016
// OUTPUT: 1048064
// OUTPUT: 983552
// EXIT: 0

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    int data[512];
    int chk0 = 0;
    int chk1 = 0;
    int chk2 = 0;
    int reps = 127 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 29) & 127;

        for (int i = 0; i < 512; i = i + 1) {
            data[i] = ((i * 37 + seed) & 255) - 128;
        }

        int pos = 0;
        int neg = 0;
        int clamped = 0;
        for (int pass = 0; pass < 4; pass = pass + 1) {
            for (int i = 0; i < 512; i = i + 1) {
                int v = data[i];
                if (v > 0) {
                    pos = pos + v;
                } else {
                    neg = neg - v;
                }
                if (v > 64) {
                    clamped = clamped + 64;
                } else {
                    if (v < -64) {
                        clamped = clamped - 64;
                    } else {
                        clamped = clamped + v;
                    }
                }
            }
        }

        chk0 = (chk0 + (pos)) & 1048575;
        chk1 = (chk1 + (neg)) & 1048575;
        chk2 = (chk2 + (clamped)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    printf("%d\n", chk2);
    return 0;
}
