// KNOWN FAILING -- wrong value at -O0. Do not "fix" by weakening it.
//
//   cc -O0 and cc -O2   101
//   blitz -O0            67
//   blitz -O1            cannot allocate registers (gpr_overshoot=1)
//
// Both verifiers are silent at every level, so this is a value error and not an
// absence: every register is written before it is read and no slot is reloaded
// unwritten.
//
// FIVE WRONG TERMS, named by `tests/fuzz/perturb.py --opt=-O0` in one run:
//
//   term     cc delta   blitz delta
//   v7          1000             0
//   v10            0            34
//   v12         1000          1034
//   arr[6]      1000          1034
//   arr[7]      1000             0
//
// The shape to start from: two terms do not respond to their own initialiser at
// all while three pick up the same +34, which reads as several terms resolving to
// one value that is not theirs. 34 is also the whole discrepancy (101 - 67).
//
// This is the 87-line reduction of gen_c.py seed 18, shape pressure, whose
// unreduced form is now tests/lit/regalloc/coalesce_pair_from_schedule.c. It is a
// DIFFERENT bug: the reduction prints 67 both before and after the copy-pair fix
// that made the unreduced program correct, which is why that test is kept
// unreduced.
//
// EXIT: 0
// OUTPUT: 101
extern int printf(char* fmt, int x);
double f0(double p0, double p1, double p2, int p3, double p4, int p5, int p6) {
    return (((p3 & 1023) & 768) & 63);
}
int f1(int p0, double p1, int p2, int p3, double p4, int p5, double p6, int p7) {
    if (((p7 * p0) >= (p0 % 2))) {
        if ((p3 <= -24)) {
        }
        for (int i44 = 0; i44 < 1; i44++) {
            if ((p2 < -99)) {
            }
        }
        for (int i84 = 0; i84 < 5; i84++) {
        }
    }
    return ((p2 / 2) / -6);
}
double f2(double p0, double p1, double p2, double p3, int p4, int p5, double p6, int p7, int p8) {
    for (int i19 = 0; i19 < 6; i19++) {
    }
    return ((p6 - p2) - (((p8 & 255) << 3) & 63));
}
int main() {
    int arr[8];
    arr[0] = -7;
    arr[1] = -4;
    arr[2] = -1;
    arr[3] = 2;
    arr[5] = 8;
    arr[6] = 11;
    arr[7] = 14;
    int v0 = 41;
    int v1 = 5;
    int v2 = -29;
    int v3 = 27;
    int v4 = 26;
    int v5 = -44;
    int v6 = 25;
    int v7 = 21;
    int v8 = -33;
    int v9 = 1;
    int v10 = -41;
    int v11 = 23;
    int v12 = 4;
    int v13 = -3;
    int v14 = 37;
    int v15 = -20;
    double d0 = 14.0;
    double d1 = 8.0;
    double d2 = -25.0;
    double d3 = -15.0;
    double d4 = -22.0;
    double d5 = -6.0;
    double d6 = -22.0;
    double d7 = 17.0;
    double d8 = 4.0;
    double d9 = -3.0;
    double d10 = -27.0;
    double d11 = -24.0;
    for (int i99 = 0; i99 < 2; i99++) {
        if ((((87 & 255) >> 3) < 45)) {
            if ((85 <= v10)) {
                if ((v2 == 27)) {
                    if ((88 != v10)) {
                    }
                }
            }
            v10 = i99;
        } else {
            if ((-41 <= i99)) {
            }
        }
        arr[((v7 - v9)) & 7] = arr[(((-80 & 255) << 4)) & 7];
    }
    v6 = f1((36 / -6), d10, 98, v15, d7, (-51 - v12), 38.0, ((-100 & 255) >> 1));
    if ((v1 > (((v6 % 7) & 255) << 1))) {
    }
    d8 = f2((d9 + d2), (35.0 - d1), (((-27 & 1023) ^ 142) & 63), ((98 + 95) & 63), (v11 / 11), ((v11 & 1023) ^ 570), -2.0, ((v4 & 1023) | 63), ((11 & 255) << 4));
    for (int i36 = 0; i36 < 4; i36++) {
        if ((v2 <= (-11 / 2))) {
            v0 = ((v4 & 1023) & 514);
            v15 = 51;
        }
    }
    v13 = v9;
    printf("%d\n", (((((((((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + v10) + v11) + v12) + v13) + v14) + v15) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
