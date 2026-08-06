// A hash chain: multiply, xor and shift, each iteration feeding the next.
//
// This is the latency-bound counterpart to `accum_pair`. Nothing in the body
// can be overlapped with anything else, so the emitted code is judged on the
// length of the chain alone: an `imul` that has to be built out of `lea`s, a
// shift that costs a copy because the count is not in `cl`, or a spill anywhere
// on the chain all land on the critical path directly. Unsigned throughout, so
// the wraparound is defined and the answer is the same on any compiler.

// OUTPUT: 1391309969
// OUTPUT: 3289033470
// OUTPUT: 2071
// EXIT: 0

extern int printf(char* fmt, unsigned int x);

int main(int argc, char** argv) {
    unsigned int tbl[128];
    unsigned int seed = (unsigned int)((argc * 7919) & 65535);

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

    printf("%u\n", h);
    printf("%u\n", g);
    printf("%u\n", odd);
    return 0;
}
