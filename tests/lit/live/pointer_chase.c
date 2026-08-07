// A dependent chain of loads: each element holds the index of the next, so no
// load can issue until the previous one lands.
//
// Nothing here can be hoisted, unrolled into independent work, or folded. What
// it measures is the cost of the address computation and the load itself, which
// is what addressing-mode selection is for.

// OUTPUT: 846848
// OUTPUT: 0
// EXIT: 0

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    int next[256];
    int payload[256];
    int chk0 = 0;
    int chk1 = 0;
    int reps = 66 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int step = (((argc + r) * 5) & 7) + 3;

        for (int i = 0; i < 256; i = i + 1) {
            next[i] = (i + step) & 255;
            payload[i] = (i * 13) & 1023;
        }

        int at = 0;
        int acc = 0;
        for (int hop = 0; hop < 2048; hop = hop + 1) {
            acc = acc + payload[at];
            at = next[at];
        }

        chk0 = (chk0 + (acc)) & 1048575;
        chk1 = (chk1 + (at)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
