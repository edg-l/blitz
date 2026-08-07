// Fifteen parameters, and only the first six of them hold a register at entry.
//
// A parameter that is already in a register before the block's first instruction
// runs has to keep one over the whole run of parameter markers, whatever order
// the scheduler put them in -- a block parameter because the phi copies on the
// edge wrote it, and a register-passed parameter because the caller did. So all
// of those mutually interfere.
//
// A stack-passed parameter is not one of them: its value is in the caller's
// frame and its marker lowers to the load that fetches it. Counted as resident,
// all fifteen parameters here formed a clique of fifteen where fourteen colours
// exist, and `callee` did not compile at `-O1` -- with a measured peak of eight
// GPRs live, so the graph and the pressure disagreed by seven.
//
// Fourteen parameters fitted exactly and hid this, which is why the test takes
// fifteen.

extern int printf(char* fmt, ...);

__attribute__((noinline)) int fifteen(int p0, int p1, int p2, int p3, int p4,
                                      int p5, int p6, int p7, int p8, int p9,
                                      int p10, int p11, int p12, int p13,
                                      int p14) {
    if (p0 != 200) { return 1; }
    if (p1 != 201) { return 2; }
    if (p2 != 202) { return 3; }
    if (p3 != 203) { return 4; }
    if (p4 != 204) { return 5; }
    if (p5 != 205) { return 6; }
    if (p6 != 206) { return 7; }
    if (p7 != 207) { return 8; }
    if (p8 != 208) { return 9; }
    if (p9 != 209) { return 10; }
    if (p10 != 210) { return 11; }
    if (p11 != 211) { return 12; }
    if (p12 != 212) { return 13; }
    if (p13 != 213) { return 14; }
    if (p14 != 214) { return 15; }
    return 0;
}

int main() {
    printf("%d\n", fifteen(200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                           210, 211, 212, 213, 214));
    return 0;
}

// EXIT: 0
// OUTPUT: 0
