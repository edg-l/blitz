// An inlinable callee at a call site the register file cannot afford.
//
// The counterpart to `call_inlinable.c`, and the case a pressure check on the
// inliner exists for: the loop already carries more live values than there are
// registers, so inlining the callee's own values into it buys the ABI cost back
// and pays for it in spills. `call_inlinable.c` is the same decision where the
// answer goes the other way, and a check that gets one right and the other wrong
// is not a check.
//
// The twelve accumulators are all live across the call and all read after it, so
// none can be rematerialized and none is dead. The callee is deliberately larger
// than `mix`: four locals of its own, so inlining adds to the pressure rather
// than merely moving the ABI cost.

// OUTPUT: 172339
// OUTPUT: 136285
// EXIT: 0

extern int printf(char* fmt, ...);

int blend(int a, int b, int c, int d) {
    int p = (a * 5 + b) & 2047;
    int q = (c * 3 + d) & 2047;
    int r = (p ^ q) + ((p + q) & 511);
    int s = (r * 7) & 1023;
    if (s > p) {
        s = s - (q & 255);
    } else {
        s = s + (p & 255);
    }
    return (r + s) & 4095;
}

int main(int argc, char** argv) {
    int chk0 = 0;
    int chk1 = 0;
    int reps = 47 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 29) & 127;
        int a0 = seed + 1;
        int a1 = seed + 2;
        int a2 = seed + 3;
        int a3 = seed + 5;
        int a4 = seed + 7;
        int a5 = seed + 11;
        int a6 = seed + 13;
        int a7 = seed + 17;
        int a8 = seed + 19;
        int a9 = seed + 23;
        int a10 = seed + 29;
        int a11 = seed + 31;
        for (int i = 0; i < 512; i = i + 1) {
            int v = blend(a0 + i, a1, a2 + (i & 15), a3);
            a0 = (a0 + v) & 8191;
            a1 = (a1 ^ (v + a2)) & 8191;
            a2 = (a2 + (v & 255)) & 8191;
            a3 = (a3 + (a4 & 127)) & 8191;
            a4 = (a4 + (a5 & 127)) & 8191;
            a5 = (a5 + (a6 & 127)) & 8191;
            a6 = (a6 + (a7 & 127)) & 8191;
            a7 = (a7 + (a8 & 127)) & 8191;
            a8 = (a8 + (a9 & 127)) & 8191;
            a9 = (a9 + (a10 & 127)) & 8191;
            a10 = (a10 + (a11 & 127)) & 8191;
            a11 = (a11 + (a0 & 127)) & 8191;
        }
        chk0 = (chk0 + a0 + a2 + a4 + a6 + a8 + a10) & 1048575;
        chk1 = (chk1 + a1 + a3 + a5 + a7 + a9 + a11) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
