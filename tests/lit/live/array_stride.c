// A strided walk over an array whose contents the compiler cannot know.
//
// The address `base + i*stride*4` is recomputed from scratch every iteration
// and the bound is loop-invariant, so this is where loop-invariant motion and
// strength reduction have to show. `argc` seeds the data: without it a
// reference compiler evaluates the whole program and prints the constant, and
// the comparison measures nothing.

// OUTPUT: 821335
// OUTPUT: 783104
// EXIT: 0

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    int arr[512];
    int chk0 = 0;
    int chk1 = 0;
    int reps = 538 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 37) & 63;

        for (int i = 0; i < 512; i = i + 1) {
            arr[i] = ((i * 7) + seed) & 255;
        }

        int total = 0;
        for (int stride = 1; stride < 5; stride = stride + 1) {
            int i = 0;
            while (i < 512) {
                total = total + arr[i];
                i = i + stride;
            }
        }

        int evens = 0;
        for (int i = 0; i < 512; i = i + 2) {
            evens = evens + arr[i];
        }

        chk0 = (chk0 + (total)) & 1048575;
        chk1 = (chk1 + (evens)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
