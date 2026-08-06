// A loop whose branch outcome depends on data the compiler cannot know.
//
// Nothing here folds into straight-line code, so the block layout, the
// comparison and the conditional jump are all real. This is the shape where
// branch layout and `cmov`-versus-branch selection decide the answer.

// OUTPUT: 65024
// OUTPUT: 66048
// OUTPUT: -512
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int data[512];
    int seed = (argc * 29) & 127;

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

    printf("%d\n", pos);
    printf("%d\n", neg);
    printf("%d\n", clamped);
    return 0;
}
