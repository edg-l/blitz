// Regression test: one argument of an edge was unhooked because another argument
// on the same edge shared its VReg.
//
// Was KNOWN FAILING -- cc said 106, blitz -O0 said 82 -- and exactly one term of
// the sum was wrong, v14 arriving as -3 where 21 belongs, which is the whole 24 of
// the discrepancy.
//
// Block 16's terminator passed one VReg at BOTH argument position 0 and position
// 15, because its target has two parameters of the same e-class. Position 0's
// destination parameter was routed through a stack slot, position 15's was not,
// and `remove_terminator_arg_operands` dropped operands by VReg -- so position 15
// lost its operand too. Nothing then wrote that parameter on this edge, and the
// target block read whatever the register happened to hold: -3, another variable's
// initialiser.
//
// At -O1 the same function is also where slot routing takes a parameter that
// names the value it carries: three of block 30's parameters share block 0's
// VRegs, and the edge feeding them is the one edge that must store to their
// slots. Read as a back edge instead, the slots stay unwritten and the sum is
// 117.
//
// 88 lines, reduced from gen_c.py seed 18, shape pressure. The reducer's output had
// lost `arr[4] = 5;` while the sum still reads arr[4], an uninitialised read; it is
// restored here and the divergence survived it.
//
// EXIT: 0
// OUTPUT: 106
extern int printf(char* fmt, int x);
double f0(double p0, double p1, double p2, int p3, double p4, int p5, int p6) {
    return (((p3 & 1023) & 768) & 63);
}
int f1(int p0, double p1, int p2, int p3, double p4, int p5, double p6, int p7) {
    if (((p7 * p0) >= (p0 % 2))) {
        if ((p3 <= -24)) {
        }
    }
    if (((p7 / 8) <= p2)) {
        for (int i44 = 0; i44 < 1; i44++) {
            if ((p2 < -99)) {
            }
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
        v11 = (((v13 / 7) & 1023) | 11);
        if ((((87 & 255) >> 3) < 45)) {
            if ((85 <= v10)) {
                if ((v2 == 27)) {
                    if ((88 != v10)) {
                    }
                }
            }
            v10 = i99;
            v7 = (-12 - v8);
            if ((-41 <= i99)) {
                v14 = v7;
            }
        }
        arr[((v7 - v9)) & 7] = arr[(((-80 & 255) << 4)) & 7];
    }
    if ((v1 > (((v6 % 7) & 255) << 1))) {
    }
    for (int i36 = 0; i36 < 4; i36++) {
        if ((v2 <= (-11 / 2))) {
            v0 = ((v4 & 1023) & 514);
            v15 = 51;
        }
    }
    v13 = v9;
    printf("%d\n", (((((((((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + v10) + v11) + v12) + v13) + v14) + v15) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
