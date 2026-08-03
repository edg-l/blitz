// Regression test: two parameters of one block are the same e-class, so the
// parallel copy on the edge into it named one register twice.
//
// `propagate_block_params` merges a block parameter with its incoming argument
// when the block has a single predecessor and that argument is a constant. Here
// `main` jumps to a block passing the constant 0 for two different parameters,
// so both parameters merge onto that constant's class. One class is one VReg is
// one register, and that is correct -- a single predecessor passing the same
// constant twice means the two parameters provably hold the same value.
//
// Each argument is its own operand of the terminator, though, and the constant
// is materialized into a different register per operand. The edge therefore
// asked for [(RDX, R14), (RCX, R14), (RAX, R9), (R10, RSI)] and aborted in
// `sequentialize_copies`: a parallel copy cannot express two writes to one
// register. Both copies say the same thing, so one of them carries the value and
// the other is dropped.
//
// OUTPUT: -780
// EXIT: 0
extern int printf(char* fmt, int x);
double f0(int p0, double p1, double p2, int p3, int p4, int p5, double p6) {
    if ((((p5 & 255) << 4) <= p0)) {
        if ((p0 <= p0)) {
            if ((-87 <= 97)) {
                if ((-76 >= p5)) {
                }
            }
        }
    } else {
        for (int i10 = 0; i10 < 2; i10++) {
            if ((i10 >= 64)) {
                if ((p0 != i10)) {
                    if ((p4 > -60)) {
                    }
                }
            }
        }
    }
    if ((42 <= p4)) {
        if ((p4 >= p5)) {
            if ((44 >= 49)) {
            }
            if ((p0 != p3)) {
                for (int i32 = 0; i32 < 3; i32++) {
                }
            }
        }
    }
    return ((p6 + -31.0) - (40.0 + p2));
}
int main() {
    int arr[8];
    arr[1] = -4;
    arr[2] = -1;
    arr[3] = 2;
    arr[4] = 5;
    arr[6] = 11;
    arr[7] = 14;
    int v0 = 31;
    int v1 = -10;
    int v2 = 21;
    int v3 = 32;
    int v4 = 50;
    int v5 = -22;
    int v6 = -46;
    int v7 = -20;
    int v8 = 13;
    int v9 = 47;
    double d0 = 14.0;
    double d3 = -13.0;
    double d4 = -27.0;
    double d5 = -28.0;
    if ((((((v1 & 1023) & 876) & 1023) | 581) > (((v1 + v0) & 255) << 1))) {
        arr[((v3 % 16)) & 7] = (v6 % 7);
        for (int i53 = 0; i53 < 2; i53++) {
            arr[(v2) & 7] = (-29 * v0);
        }
    }
    d0 = f0(0, (d3 + d5), (d4 + d4), v0, ((v8 & 1023) | 130), v8, (d0 - d5));
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
