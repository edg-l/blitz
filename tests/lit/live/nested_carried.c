// Nested loops whose inner body carries a dependence across iterations, over
// data seeded at runtime.
//
// `grid[i]` reads `grid[i-1]` that the previous iteration just wrote, and the
// outer pass feeds `carry` from the last element back into the next pass. So
// nothing here can be vectorized, unrolled into independent chains, or hoisted:
// what is left is address arithmetic, one load and one store per iteration, and
// a value that has to stay live across the whole inner loop.

// OUTPUT: 147352
// OUTPUT: 1912
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int grid[64];
    int chk0 = 0;
    int chk1 = 0;
    int reps = 137 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 7) & 15;

        for (int i = 0; i < 64; i = i + 1) {
            grid[i] = (i * 3 + seed) & 63;
        }

        int carry = seed + 1;
        for (int pass = 0; pass < 32; pass = pass + 1) {
            for (int i = 1; i < 64; i = i + 1) {
                grid[i] = (grid[i] + grid[i - 1] + carry) & 1023;
            }
            carry = (carry + grid[63]) & 31;
        }

        int sum = 0;
        for (int i = 0; i < 64; i = i + 1) {
            sum = sum + grid[i];
        }

        chk0 = (chk0 + (sum)) & 1048575;
        chk1 = (chk1 + (carry)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
