// A 32x32 transpose accumulated over four passes: the read walks one row
// contiguously while the write strides by the row length, with the data seeded
// at runtime.
//
// The two arrays are distinct objects, so `alias.rs` may keep `a[i*32+j]` alive
// across the write to `b[j*32+i]`; the strided write is what stops the address
// arithmetic from collapsing to a single induction variable, so this is where
// LICM's hoisting of `j * 32` and the addressing-mode scale have to pay.

// OUTPUT: 20480
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int a[1024];
    int b[1024];
    int seed = (argc * 7) & 15;

    for (int i = 0; i < 32; i = i + 1) {
        for (int j = 0; j < 32; j = j + 1) {
            a[i * 32 + j] = ((i * 13 + j * 7 + seed) & 255) - 128;
            b[i * 32 + j] = 0;
        }
    }

    for (int pass = 0; pass < 4; pass = pass + 1) {
        for (int i = 0; i < 32; i = i + 1) {
            for (int j = 0; j < 32; j = j + 1) {
                b[j * 32 + i] = b[j * 32 + i] + a[i * 32 + j];
            }
        }
    }

    int sum = 0;
    for (int i = 0; i < 1024; i = i + 1) {
        sum = sum + b[i] * ((i & 3) + 1);
    }
    printf("%d\n", sum);
    return 0;
}
