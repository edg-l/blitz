// Bit manipulation over runtime words: population count the slow way, and
// Kernighan's clear-lowest-set-bit loop.
//
// Every one of these is a single x86 instruction the backend does not yet
// select (`popcnt`, `bsf`, `blsr`), so this kernel is the yardstick for P2's
// bit-instruction item -- and it has to run on data the compiler cannot fold.

// OUTPUT: 1918
// OUTPUT: 1918
// OUTPUT: 2560
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int words[256];
    int seed = (argc * 2654) & 65535;

    for (int i = 0; i < 256; i = i + 1) {
        words[i] = (i * 2654 + seed) & 65535;
    }

    int bits = 0;
    for (int i = 0; i < 256; i = i + 1) {
        int w = words[i];
        for (int b = 0; b < 16; b = b + 1) {
            bits = bits + ((w >> b) & 1);
        }
    }

    int kern = 0;
    for (int i = 0; i < 256; i = i + 1) {
        int w = words[i];
        while (w != 0) {
            w = w & (w - 1);
            kern = kern + 1;
        }
    }

    int lowest = 0;
    for (int i = 0; i < 256; i = i + 1) {
        int w = words[i];
        if (w != 0) {
            lowest = lowest + (w & (0 - w));
            lowest = lowest & 1048575;
        }
    }

    printf("%d\n", bits);
    printf("%d\n", kern);
    printf("%d\n", lowest);
    return 0;
}
