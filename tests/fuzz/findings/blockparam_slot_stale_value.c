// KNOWN FAILING -- wrong value at -O0. Do not "fix" by weakening it.
//
//   cc -O0 and cc -O2   106
//   blitz -O0            82
//   blitz -O1            cannot allocate registers (gpr_overshoot=3)
//
// The -O1 refusal arrived with aa91e96, which stopped routing a block parameter
// that is really a value. Before it, -O1 emitted code that read three slots
// nothing had written.
//
// Both verifiers are silent on it: the machine verifier sees every register
// written before it is read, and the register-sharing check sees no two live
// values in one register. It is a value error, not an absence.
//
// ONE TERM OF THE SUM IS WRONG: v14. `read_frame.py --sum-chain` breaks at every
// add of the printf's chain in the unmodified binary and prints the term and the
// running total; v14 arrives as -3 where 21 (`v14 = v7`) is right, and -24 is the
// whole discrepancy, 106 - 82, so no other term is involved.
//
// WHERE IT IS, traced step by step off the running binary with read_frame.py.
// Each step is a fact, not an inference:
//
//   v14's term            <- slot 62, reloaded
//   slot 62               <- b1's p15 (v60), spilled every iteration; 37, -3, -3
//   b1's p15 on the latch <- b3's arg, SpillLoad(53)
//   slot 53               <- v321 = BlockParam(19, 15), spilled after its def
//   v321                  <- `mov r15d,eax`, the b18 -> 19 copy, EAX = -3
//   EAX                   <- b18's p15, written by the b17 -> 18 copy from RBP
//   RBP                   <- v384, class 268, already -3 there
//
// So the wrong value is already wrong when block 17 hands it over, and the search
// continues above that: what is v384, and which parameter position is supposed to
// carry v7 on this path?
//
// The interesting fact at block 17 is that it DOES compute v7 freshly --
// `v276 = Iconst(-12)`, `v277 = X86Sub([276, 260])`, `v278 = Proj0([277])`, giving
// 21 -- and its terminator passes v278 at argument position 0, while position 15,
// the one that reaches v14, carries v384 instead. `v14 = v7` should put v7's class
// in both places.
//
// RULED OUT, and it looked convincing for a while: an illegal coalesce. The phi
// trace prints b18's p15 as VReg 278 and v278 holds RAX, which reads as a block
// parameter sharing a register with a live value. It is not: the trace chases
// coalesce aliases before printing, v278 is merely the representative of
// `v297 = BlockParam(18, 15)`, and v278's own range is entirely inside block 17
// (defined at [2], consumed by the terminator at [3], live in zero blocks per the
// allocator). The merge is legal and RAX legitimately holds the parameter.
//
// Worth keeping from that dead end: `verify_register_sharing` canonicalizes every
// VReg through the coalesce aliases before it counts, so if a merge ever IS
// illegal this check cannot see it -- the two values have become one VReg by the
// time it looks. That blind spot is real whatever this bug turns out to be.
//
// `perturb.py` flags four terms (v8, v10, v13, arr[6]) and none is the fault. That
// is the tool's limit, not a contradiction: it perturbs an initialiser, and a
// changed constant folds differently downstream, so near the allocator's limit the
// probe moves the bug. Confirm every hit against the unmodified binary.
//
// It needs the block-parameter slot routing to reproduce: with routing disabled
// the program does not allocate at either level, and neither does it at 2f25de1,
// before routing existed. So the wrong value is either in that path or was made
// reachable by it. 17 parameters are routed at -O0.
//
// Reduced from gen_c.py seed 18, shape pressure, 192 -> 88 lines. The reducer's
// output had lost `arr[4] = 5;` while the sum still reads arr[4], which is an
// uninitialised read; it is restored here and the divergence survives it.
//
// Promote to tests/lit once blitz prints 106 at both levels.
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
