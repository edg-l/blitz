// Unsigned and mixed-width arithmetic over runtime data: bytes widened to
// unsigned int, signed shorts accumulated into an int, and an unsigned compare
// that a signed one would get wrong.
//
// Every load here is a widening one (`movzx` from a byte, `movsx` from a half)
// and the two accumulators have different signedness, so a lost or wrong-width
// extension changes the answer rather than the instruction count.

// OUTPUT: 388608
// OUTPUT: 1833472
// OUTPUT: 648
// EXIT: 0

extern int printf(char* fmt, unsigned int x);

int main(int argc, char** argv) {
    unsigned char bytes[256];
    short halves[128];
    unsigned int seed = (unsigned int)((argc * 91) & 255);

    for (int i = 0; i < 256; i = i + 1) {
        bytes[i] = (unsigned char)(((unsigned int)i * 17 + seed) & 255);
    }
    for (int i = 0; i < 128; i = i + 1) {
        halves[i] = (short)(((i * 313 + (int)seed) & 32767) - 16384);
    }

    unsigned int usum = 0;
    int ssum = 0;
    unsigned int wide = 0;
    for (int pass = 0; pass < 8; pass = pass + 1) {
        for (int i = 0; i < 128; i = i + 1) {
            usum = (usum + (unsigned int)bytes[i] * 3) & 1048575;
            ssum = (ssum + (int)halves[i]) & 2097151;
            if ((unsigned int)bytes[i] > (unsigned int)(seed & 127)) {
                wide = (wide + 1) & 4095;
            }
        }
    }

    printf("%u\n", usum);
    printf("%u\n", (unsigned int)ssum);
    printf("%u\n", wide);
    return 0;
}
