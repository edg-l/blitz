// Regression test: a value that lived and died before a block parameter's own
// `BlockParam` marker took the register that parameter was holding.
//
// The phi copies on the edge write every parameter of a block before its first
// instruction runs; a marker only says where the value is named, and the
// scheduler places the markers by dependence order. Nothing modelled that, so a
// splitter store/reload pair inserted inside the run of markers was given a
// register a later-marked parameter already held. Both verifiers pass on it --
// the register is written before it is read, and no two *modelled* live ranges
// overlap.
//
// Was KNOWN FAILING: cc said 101 and blitz -O0 said 72, from ONE wrong value.
// `v7` is the constant 21, never assigned again, and blitz had 1 in it -- `v9`,
// the constant beside it. The loop's only store indexes on it,
// `arr[((v7 - v9)) & 7] = arr[0]`, so blitz wrote arr[7] where arr[4] belongs:
// three wrong terms (v7 -20, arr[4] +12, arr[7] -21) from one register.
//
// Located by reading the sum off the unmodified binary with
// `read_frame.py --sum-chain` against a `cc` build whose printf is split into
// one call per term. `perturb.py` alone named five terms on this program, two of
// them artefacts of the probe: it sits at the allocator's limit, so changing a
// constant moves the allocation.
//
// Pinned at -O0 because -O1 still cannot allocate this function, which is
// register pressure and tracked separately. The opt-level pin excludes it from
// the differential harness, by design.
//
// 72 lines, reduced from gen_c.py seed 18 shape pressure. Both reduction passes
// deleted `arr[4] = 5;` while the sum still reads arr[4]; reduce.py's UB guard
// cannot see it, because the loop writes arr through a computed index. Check for
// that before trusting any further reduction of a program with an array.
//
// EXIT: 0
// OUTPUT: 101
// FLAGS: -O0
extern int printf(char* fmt, ...);
double f0(double p0, double p1, double p2, int p3, double p4, int p5, int p6) {
    return (((p3 & 1023) & 768) & 63);
}
int f1(int p0, double p1, int p2, int p3, double p4, int p5, double p6, int p7) {
    if (((p7 * p0) >= (p0 % 2))) {
        for (int i44 = 0; i44 < 1; i44++) {
            if ((p2 < -99)) {
            }
        }
    }
    return ((p2 / 2) / -6);
}
double f2(double p0, double p1, double p2, double p3, int p4, int p5, double p6, int p7, int p8) {
    return ((p6 - p2) - (((p8 & 255) << 3) & 63));
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
    double d1 = 8.0;
    double d2 = -25.0;
    double d7 = 17.0;
    double d8 = 4.0;
    double d9 = -3.0;
    double d10 = -27.0;
    for (int i99 = 0; i99 < 2; i99++) {
        if ((((87 & 255) >> 3) < 45)) {
            if ((85 <= v10)) {
                if ((v2 == 27)) {
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
