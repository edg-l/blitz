// A two-lane mixing loop whose body is a rotate and a funnel shift.
//
// The shapes isel folds into `rol` and `shld` are three instructions each when
// they are not folded, and both sit on the loop's carried dependence, so a miss
// costs the whole chain rather than a byte. The two lanes also keep four values
// live across the body, which is what makes the funnel shift's single read of
// each operand worth more than its latency: the shift pair would copy both.
//
// Seeded from argc, so no reference compiler can evaluate the loop.

// OUTPUT: 3422184273
// OUTPUT: 1959448722
// OUTPUT: 2089738
// EXIT: 0

extern int printf(char* fmt, unsigned int x);

int main(int argc, char** argv) {
    unsigned int buf[64];
    unsigned int seed;
    unsigned int hi;
    unsigned int lo;
    unsigned int acc;
    unsigned int mixed;
    int pass;
    int i;

    seed = (unsigned int)((argc * 2654435761) & 65535) | 1;
    for (i = 0; i < 64; i = i + 1) {
        buf[i] = (unsigned int)i * 2246822519 + seed;
    }

    hi = seed;
    lo = seed ^ 3735928559;
    acc = 0;
    for (pass = 0; pass < 64; pass = pass + 1) {
        for (i = 0; i < 64; i = i + 1) {
            hi = (hi << 7) | (hi >> 25);
            mixed = (hi << 11) | (lo >> 21);
            lo = (lo + buf[i]) ^ mixed;
            hi = hi ^ (lo >> 3);
            acc = acc + (mixed & 1023);
        }
    }

    printf("%u\n", hi);
    printf("%u\n", lo);
    printf("%u\n", acc);
    return 0;
}
