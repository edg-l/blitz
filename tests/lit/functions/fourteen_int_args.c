// A call with more arguments than the machine has registers.
//
// Fourteen integers: six in RDI..R9 and eight pushed. The call barrier reads
// every argument at one program point, so the colouring needed fourteen GPRs at
// one instruction and there are thirteen -- and no amount of spilling relieves a
// point where the instruction itself is what reads the values. But a pushed
// argument is only going to be pushed, and a push can read memory, so those
// eight need no register at all: `allocate_global` routes them through frame
// slots once a round has failed, and `abi::setup_call_args` pushes them out of
// those slots.
//
// The callee returns the 1-based index of the first argument that did not
// arrive, so a failure names the argument rather than merely disagreeing.

extern int printf(char* fmt, ...);

__attribute__((noinline)) int fourteen(int p0, int p1, int p2, int p3, int p4,
                                       int p5, int p6, int p7, int p8, int p9,
                                       int p10, int p11, int p12, int p13) {
    if (p0 != 100) { return 1; }
    if (p1 != 101) { return 2; }
    if (p2 != 102) { return 3; }
    if (p3 != 103) { return 4; }
    if (p4 != 104) { return 5; }
    if (p5 != 105) { return 6; }
    if (p6 != 106) { return 7; }
    if (p7 != 107) { return 8; }
    if (p8 != 108) { return 9; }
    if (p9 != 109) { return 10; }
    if (p10 != 110) { return 11; }
    if (p11 != 111) { return 12; }
    if (p12 != 112) { return 13; }
    if (p13 != 113) { return 14; }
    return 0;
}

int main() {
    printf("%d\n", fourteen(100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                            110, 111, 112, 113));
    return 0;
}

// EXIT: 0
// OUTPUT: 0
