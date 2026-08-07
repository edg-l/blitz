// A triple loop nest over a flat buffer indexed by hand.
//
// `buf[(i * 12 + j) * 10 + k]` is a multiply chain the addressing mode can
// absorb and an induction variable the strength reducer should turn into an
// add. Written flat rather than as `buf[i][j][k]` so the arithmetic is in the
// source, not in the subscript rules.

// OUTPUT: 1287
// OUTPUT: 902880
// EXIT: 0


extern int printf(char* fmt, ...);

int main() {
    int buf[1440];
    for (int i = 0; i < 1440; i = i + 1) {
        buf[i] = 0;
    }

    for (int i = 0; i < 12; i = i + 1) {
        for (int j = 0; j < 12; j = j + 1) {
            int row = (i * 12 + j) * 10;
            for (int k = 0; k < 10; k = k + 1) {
                buf[row + k] = buf[row + k] + i * 100 + j * 10 + k;
            }
        }
    }

    // A second nest whose invariant is only invariant in the innermost loop.
    for (int i = 0; i < 12; i = i + 1) {
        int scale = i * 3 + 1;
        for (int j = 0; j < 12; j = j + 1) {
            for (int k = 0; k < 10; k = k + 1) {
                int idx = (i * 12 + j) * 10 + k;
                buf[idx] = buf[idx] + scale * (j - k);
            }
        }
    }

    int total = 0;
    int corner = buf[1439];
    for (int i = 0; i < 1440; i = i + 1) {
        total = total + buf[i];
    }
    printf("%d\n", corner);
    printf("%d\n", total);
    return 0;
}
