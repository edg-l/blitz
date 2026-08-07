// A small dense matrix multiply whose inputs the compiler cannot see.
//
// Three nested loops over one flat array each: the inner loop's address
// arithmetic is `base + (i*n + k)*4`, of which only the `k` term varies. What
// the invariant part costs is the whole point of loop-invariant motion and
// induction-variable work.

// OUTPUT: 398848
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int a[576];
    int b[576];
    int c[576];
    int chk0 = 0;
    int reps = 68 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int n = 24;
        int seed = ((argc + r) * 5) & 15;

        for (int i = 0; i < 576; i = i + 1) {
            a[i] = (i + seed) & 31;
            b[i] = (i * 3 + 1) & 31;
            c[i] = 0;
        }

        for (int i = 0; i < n; i = i + 1) {
            for (int j = 0; j < n; j = j + 1) {
                int sum = 0;
                for (int k = 0; k < n; k = k + 1) {
                    sum = sum + a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum & 65535;
            }
        }

        int total = 0;
        for (int i = 0; i < 576; i = i + 1) {
            total = (total + c[i]) & 1048575;
        }
        chk0 = (chk0 + (total)) & 1048575;
    }
    printf("%d\n", chk0);
    return 0;
}
