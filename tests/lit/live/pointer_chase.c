// A dependent chain of loads: each element holds the index of the next, so no
// load can issue until the previous one lands.
//
// Nothing here can be hoisted, unrolled into independent work, or folded. What
// it measures is the cost of the address computation and the load itself, which
// is what addressing-mode selection is for.

// OUTPUT: 942080
// OUTPUT: 0
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int next[256];
    int payload[256];
    int step = ((argc * 5) & 7) + 3;

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

    printf("%d\n", acc);
    printf("%d\n", at);
    return 0;
}
