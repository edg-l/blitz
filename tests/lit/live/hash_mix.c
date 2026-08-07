// A hash chain: multiply, xor and shift, each iteration feeding the next.
//
// This is the latency-bound counterpart to `accum_pair`. Nothing in the body
// can be overlapped with anything else, so the emitted code is judged on the
// length of the chain alone: an `imul` that has to be built out of `lea`s, a
// shift that costs a copy because the count is not in `cl`, or a spill anywhere
// on the chain all land on the critical path directly. Unsigned throughout, so
// the wraparound is defined and the answer is the same on any compiler.

// OUTPUT: 475861700
// OUTPUT: 600857705
// OUTPUT: 67486
// EXIT: 0

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    unsigned int tbl[128];
    unsigned int chk0 = 0;
    unsigned int chk1 = 0;
    unsigned int chk2 = 0;
    int reps = 33 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        unsigned int seed = (unsigned int)(((argc + r) * 7919) & 65535);

        for (int i = 0; i < 128; i = i + 1) {
            tbl[i] = ((unsigned int)i * 1103515245 + seed) ^ ((unsigned int)i << 13);
        }

        unsigned int h = seed | 1;
        unsigned int g = 16777619;
        unsigned int odd = 0;
        for (int pass = 0; pass < 32; pass = pass + 1) {
            for (int i = 0; i < 128; i = i + 1) {
                h = h ^ tbl[i];
                h = h * 16777619;
                h = h ^ (h >> 13);
                g = (g + h) ^ (g << 7);
                if ((h & 1) != 0) {
                    odd = odd + 1;
                }
            }
        }

        chk0 = chk0 + (unsigned int)(h);
        chk1 = chk1 + (unsigned int)(g);
        chk2 = chk2 + (unsigned int)(odd);
    }
    printf("%u\n", chk0);
    printf("%u\n", chk1);
    printf("%u\n", chk2);
    return 0;
}
