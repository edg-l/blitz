// 24x24 integer matrix multiply.
//
// Three nested loops over two-dimensional subscripts: every `a[i][k]` and
// `b[k][j]` is a multiply and an add the addressing mode could absorb, and the
// row address `a[i]` is invariant in the two inner loops.

// OUTPUT: 2730
// OUTPUT: 64744
// EXIT: 0


extern int printf(char* fmt, int x);

int main() {
    int a[24][24];
    int b[24][24];
    int c[24][24];

    for (int i = 0; i < 24; i = i + 1) {
        for (int j = 0; j < 24; j = j + 1) {
            a[i][j] = (i + j) % 7;
            b[i][j] = (i * j) % 5;
            c[i][j] = 0;
        }
    }

    for (int i = 0; i < 24; i = i + 1) {
        for (int j = 0; j < 24; j = j + 1) {
            int sum = 0;
            for (int k = 0; k < 24; k = k + 1) {
                sum = sum + a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }

    int trace = 0;
    int total = 0;
    for (int i = 0; i < 24; i = i + 1) {
        trace = trace + c[i][i];
        for (int j = 0; j < 24; j = j + 1) {
            total = total + c[i][j];
        }
    }
    printf("%d\n", trace);
    printf("%d\n", total);
    return 0;
}
