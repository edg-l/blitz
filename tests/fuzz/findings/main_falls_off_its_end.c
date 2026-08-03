// KNOWN FAILING under BLITZ_VERIFY only -- the answer is right, the allocation
// is not. Do not "fix" by weakening it.
//
// `main` falling off its end must return 0 (C99 5.1.2.2.3), and it now does; blitz
// used to exit 8 at -O0 and 223 at -O1, whatever happened to be in EAX. But the
// program cannot go in tests/lit, because `BLITZ_VERIFY=strict` has to stay green
// there and this trips the register-sharing verifier twice:
//
//   block 0 exit: VReg 1 and VReg 336 are both live and both hold RCX
//   block 0 before [45] StoreBarrier: VReg 0 and VReg 1 are both live and both hold RCX
//
// Same family as the seed 7 reduction (VReg 9 and VReg 1056 in RBX). Two reports
// in one function, one of them at a named instruction, make this the better place
// to start on it. Promote it to tests/lit once that is fixed.
//
// Was KNOWN FAILING -- blitz exited 8 at -O0 and 223 at -O1, whatever happened to
// be in EAX. Not a blanket missing-return: a six-line main returned 0 correctly,
// so it took this much surrounding code.
//
// `main` falls off its end, which C99 5.1.2.2.3 defines as returning 0. cc exits
// 0; blitz exits 8 at -O0 and 223 at -O1, so whatever happens to be in EAX is
// what the process returns.
//
// Not a blanket missing-return: a six-line main with one `if` returns 0 correctly.
// It takes this much surrounding code, so the implicit `return 0` is being emitted
// and then EAX is clobbered, or the return is dropped on one path through this
// CFG. Start by asking whether the exit block has a `mov eax,0` at all.
//
// Found while reducing fuzz seed 7: with the printf deleted the reducer drifted
// onto this instead, and reduce.py was right to accept it -- the program is
// well-defined and blitz gets it wrong. `-Werror=return-type` does not catch it
// because falling off the end of `main` is legal.
//
// EXIT: 0
extern int printf(char* fmt, int x);
double f0(int p0, int p1, double p2, int p3, double p4, double p5, int p6) {
    return -22.0;
}
int f1(double p0, double p1, int p2, double p3, int p4, double p5, int p6) {
    return ((-8 & 1023) & 981);
}
int f2(double p0, double p1, int p2, int p3, double p4, double p5, double p6) {
    if (((-46 * p3) > 55)) {
    }
    return -52;
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
    int v0 = 10;
    int v1 = -40;
    int v2 = 18;
    int v3 = 6;
    int v4 = 11;
    int v5 = 47;
    int v6 = 30;
    int v7 = -33;
    int v8 = -14;
    int v9 = 31;
    double d0 = -30.0;
    double d1 = 2.0;
    double d4 = 7.0;
    double d5 = 4.0;
    for (int i0 = 0; i0 < 1; i0++) {
    }
    if ((((((v2 & 1023) | 175) & 1023) | 370) <= (-94 - v9))) {
        if (((v8 - v4) >= ((v8 & 255) >> 4))) {
        }
    }
    for (int i68 = 0; i68 < 3; i68++) {
        v3 = (((v7 & 1023) ^ 514) + (v2 % -3));
        if (((v0 * v9) <= -30)) {
        } else {
            if ((v1 < v7)) {
                if ((v6 >= v4)) {
                }
            }
        }
    }
    if ((((((v7 & 255) << 2) & 255) >> 3) < (v1 % 8))) {
        if ((((v0 & 1023) ^ 734) > -57)) {
            for (int i96 = 0; i96 < 3; i96++) {
                if ((v6 == -13)) {
                }
            }
        }
        if ((((v7 & 1023) | 968) > (56 / 8))) {
            if ((-9 < v9)) {
                for (int i20 = 0; i20 < 4; i20++) {
                }
                for (int i34 = 0; i34 < 1; i34++) {
                    if ((v6 > v4)) {
                        if ((v3 != v8)) {
                        }
                    }
                }
                for (int i57 = 0; i57 < 4; i57++) {
                }
            }
            for (int i69 = 0; i69 < 5; i69++) {
            }
        }
    }
}
