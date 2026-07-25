// Regression test: the exit status carries the values a guard compares, so a
// wrong one is visible without a printf.
//
//   cc -O0, blitz -O0, blitz -O1   226
//
// blitz gave 104 at -O0 and, once it could allocate registers at all, 12 at -O1.
// Four bugs were found by reducing this one program: parameters re-emitted per
// block so only one copy carried the ABI precolor (438bdc4), phi args resolved
// through the global class map (ccc64b7), a loop latch overwriting its own block
// params, and R11 handed out to values while lowering used it as a scratch. The
// last two are what finally made it correct.
//
// tests/fuzz/gen_c.py seed 6, truncated after the first call and returning the
// values the original program's guard compares.
//
// EXIT: 226
//
extern int printf(char* fmt, int x);

int f0(double p0, double p1, double p2, double p3, int p4, int p5, int p6) {
    for (int i9 = 0; i9 < 1; i9++) {
        i9 = ((p5 & 1023) & 625);
        p4 = (p5 / -6);
        p6 = i9;
    }
    for (int i18 = 0; i18 < 6; i18++) {
        p5 = ((p6 & 1023) | 211);
    }
    return p6;
}

double f1(double p0, int p1, double p2, int p3, int p4, double p5, double p6, int p7, int p8, double p9, double p10) {
    if (((46 - p1) >= ((3 & 255) >> 1))) {
        for (int i51 = 0; i51 < 3; i51++) {
            p4 = -21;
        }
        if ((88 < p1)) {
            p3 = p3;
        } else {
            p7 = p1;
            for (int i82 = 0; i82 < 1; i82++) {
                p3 = 79;
                if ((p3 < i82)) {
                    if ((i82 < i82)) {
                        p1 = -37;
                        if ((p7 <= p1)) {
                            p3 = 42;
                        } else {
                            p7 = i82;
                        }
                    } else {
                        p4 = -24;
                        p3 = i82;
                    }
                } else {
                    p4 = i82;
                }
            }
        }
    } else {
        p3 = p1;
        p7 = ((p8 & 1023) ^ 852);
    }
    p7 = (-69 / -3);
    for (int i77 = 0; i77 < 2; i77++) {
        p8 = (p8 + p3);
    }
    return ((p6 - p6) + p0);
}

int main() {
    int arr[8];
    arr[0] = -7;
    arr[1] = -4;
    arr[2] = -1;
    arr[3] = 2;
    arr[4] = 5;
    arr[5] = 8;
    arr[6] = 11;
    arr[7] = 14;
    int v0 = -13;
    int v1 = -15;
    int v2 = -42;
    int v3 = 5;
    int v4 = -42;
    int v5 = -43;
    int v6 = 39;
    int v7 = 6;
    int v8 = -40;
    int v9 = -17;
    double d0 = 24.0;
    double d1 = 26.0;
    double d2 = 12.0;
    double d3 = -29.0;
    double d4 = -14.0;
    double d5 = -6.0;
    if (((((v4 & 1023) ^ 207) - (v7 / 8)) < ((v0 / 16) / 8))) {
        v7 = ((93 * v4) / 4);
        v6 = (((v9 % -6) & 255) >> 3);
    } else {
        v7 = ((v6 & 1023) & 671);
    }
    d5 = f1((d0 + -38.0), v2, (d4 * d1), (v5 * v8), ((-87 & 255) << 1), d1, (d0 * d1), v1, (-47 / -3), (d5 - d2), d5);
    return ((int)d5 + (int)d1 + v4) & 255;
}
