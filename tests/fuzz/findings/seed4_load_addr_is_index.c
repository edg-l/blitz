// KNOWN FAILING -- reproducer for a real bug. Do not "fix" by weakening it.
//
//   cc -O0          303
//   blitz -O0       SIGSEGV
//   blitz -O1       cannot allocate registers (see ROADMAP P0)
//
// tests/fuzz/gen_c.py seed 4. Reaches codegen only since the barrier-group pin
// rule started scanning the un-stripped schedule; before that the splitter left
// an overshoot the allocator rejected, so this never compiled.
//
// A Load resolves its address to a register holding an array INDEX:
//
//   b9 07 00 00 00    mov  ecx,0x7          ; the index 7
//   8b 31             mov  esi,DWORD PTR [rcx]   ; load from address 7
//
// So the address ClassId resolved to the VReg of the index constant. Both are
// declared as operands of the same LoadResult barrier, because
// `populate_effectful_operands` adds the folded `Addr`'s children alongside the
// address itself and then sorts the list by VReg index -- the barrier records a
// SET of VRegs, not which one fills which role. Lowering therefore has to guess
// which operand is the address, and `resolve_arg_regs_after_spilling`'s
// remat-matching case is where it guesses wrong.
//
// This is why the BLITZ_VERIFY check added alongside this file cannot catch it:
// the wrong register IS a declared barrier operand. Fixing it properly means
// giving barrier instructions role-tagged operands (address / value / arg N) so
// lowering reads the address instead of reconstructing it. See ROADMAP P0.
//
// tests/fuzz/findings/seed6_truncated_miscompile.c and seed 7 fail the same way.
//
// OUTPUT: 303
extern int printf(char* fmt, int x);

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
