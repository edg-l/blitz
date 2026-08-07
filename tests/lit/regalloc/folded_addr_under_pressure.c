// Regression test: a folded address, a cross-block spill and a block parameter
// in one high-pressure function, printing 303 at both levels.
//
// Was KNOWN FAILING three times over, each time for a different reason in the
// same seam. A Load resolved its address to the register holding its own array
// index, so it read `[rax+rax*1]`, fixed by making the leading operands of a
// barrier positional. Then a store's address register was overwritten before the
// store read it, because Phase 7 computed a second barrier-group assignment on
// the post-allocation schedule and emitted in that order instead of the one
// liveness was measured against; fixed by stopping the double grouping. Then it
// could not allocate at all -- `gpr_overshoot=2` at -O0 and 18 at -O1 -- until a
// loop-carried value could stay in a stack slot across the loop.
//
// What kept it out of tests/lit last was `BLITZ_VERIFY=strict`:
//
//   block 0 exit: VReg 20 and VReg 779 are both live and both hold RBX
//
// which was the checker's own doing. VReg 20 is the constant 0, dead where block 0
// defines it and written by the phi copy at the edge into the block that has it as
// a parameter; the check propagated successor parameters into a predecessor's
// live-out where the allocator does not.
//
// 127 lines, from gen_c.py seed 4.
//
// OUTPUT: 303
extern int printf(char* fmt, ...);

int f0(int p0, int p1, double p2, double p3, double p4, double p5, int p6, double p7, double p8, double p9, double p10, double p11) {
    if ((p1 > p6)) {
        for (int i21 = 0; i21 < 5; i21++) {
            p1 = -21;
            p6 = -83;
            p6 = i21;
        }
        p6 = ((p0 & 1023) & 653);
    } else {
        if ((p0 != p0)) {
            p0 = p6;
        } else {
            p1 = p0;
        }
    }
    for (int i20 = 0; i20 < 4; i20++) {
        p1 = (p1 - 34);
    }
    return (p1 + ((p1 & 1023) & 851));
}

double f1(double p0, int p1, double p2, int p3, double p4, double p5, int p6, double p7, double p8, int p9, double p10) {
    p6 = ((((18 & 1023) | 835) & 255) << 2);
    if ((p9 > (p9 % -3))) {
        for (int i37 = 0; i37 < 5; i37++) {
            p6 = p6;
        }
    } else {
        if ((p1 != 82)) {
            p1 = 54;
        } else {
            if ((p9 <= p3)) {
                p9 = 36;
            } else {
                p6 = p9;
                for (int i85 = 0; i85 < 3; i85++) {
                    p3 = p1;
                }
            }
            p3 = 79;
        }
        p6 = (-79 / 5);
    }
    return p4;
}

double f2(int p0, int p1, int p2, int p3, double p4, int p5, int p6, double p7) {
    p5 = ((p0 & 1023) ^ 690);
    return p4;
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
    int v1 = 21;
    int v2 = 29;
    int v3 = -19;
    int v4 = 13;
    int v5 = -11;
    int v6 = 32;
    int v7 = -41;
    int v8 = -22;
    int v9 = 13;
    double d0 = 8.0;
    double d1 = -5.0;
    double d2 = -24.0;
    double d3 = 8.0;
    double d4 = -6.0;
    double d5 = 27.0;
    for (int i38 = 0; i38 < 3; i38++) {
        v9 = ((v1 & 255) << 4);
        v4 = ((((v9 & 255) >> 3) & 1023) & 769);
        v1 = v7;
    }
    v9 = arr[(((((v5 & 255) << 3) & 255) << 3)) & 7];
    v6 = f0((-64 % 7), 85, (d1 - d5), ((v0 * v1) & 63), d4, (d1 + 30.0), v4, (-36.0 + d3), d1, -38.0, (d4 - -12.0), (26.0 * d5));
    d0 = (d0 + (26.0 * d4));
    d1 = (d1 + (d3 - -35.0));
    d2 = (d2 + ((v0 - -24) & 63));
    d3 = (d3 + d1);
    d4 = (d4 + -38.0);
    d5 = (d5 + (d4 * d3));
    arr[(((v0 & 1023) | 173)) & 7] = ((((-35 - -55) & 1023) | 455) + v4);
    if (((v0 * 58) != (((v0 & 1023) & 1001) * ((v3 & 255) >> 4)))) {
        if ((v6 >= v3)) {
            v8 = v2;
        } else {
            v5 = ((v7 & 1023) ^ 1014);
        }
        v6 = -11;
    } else {
        v7 = ((((-63 & 1023) | 190) & 255) << 1);
    }
    v4 = (v4 - v7);
    d5 = f1((d0 - -27.0), v7, (d0 - d5), 85, (37.0 - d5), (d3 * d0), (v5 - v4), -4.0, d1, v4, d0);
    d0 = f2(((v9 & 255) >> 4), ((v4 & 1023) ^ 207), ((80 & 1023) & 642), (-34 / 3), ((v6 + 11) & 63), v6, v6, -39.0);
    d0 = (d0 + (3 & 63));
    d1 = (d1 + ((v1 * v2) & 63));
    d2 = (d2 + -23.0);
    d3 = (d3 + d3);
    d4 = (d4 + (d4 + -35.0));
    d5 = (d5 + (d4 * d3));
    for (int i60 = 0; i60 < 2; i60++) {
        v7 = ((((v1 & 255) >> 0) & 255) >> 1);
        v8 = (((i60 - v9) & 1023) & 922);
        v2 = ((19 % -3) - ((v3 & 255) >> 0));
    }
    d3 = f2(11, (v1 % 4), ((v3 & 255) << 1), v9, (d4 * d4), (v1 - 83), v9, -30.0);
    d0 = (d0 + d4);
    d1 = (d1 + d4);
    d2 = (d2 + 35.0);
    d3 = (d3 + (d1 * d4));
    d4 = (d4 + ((v1 - -56) & 63));
    d5 = (d5 + (-32.0 * d3));
    if ((((((d0 + d1) + d2) + d3) + d4) + d5) != -699722.0) { return 3; }
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
    return 0;
}
