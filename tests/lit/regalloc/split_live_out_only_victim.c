// Regression test: the pressure splitter picked a victim defined in the
// over-pressure block and used only in later ones, and then had nothing to do
// with it.
//
// `main` keeps the constant 0 in a register from the top of the entry block to
// its exit; every use of it is in a successor. At the entry block's peak that
// makes it the best victim -- the longest range live there, and no local use to
// pay a reload for -- but a per-block split works by rewriting local uses, and
// with none to rewrite it planned nothing. An empty plan reads as convergence,
// so the overshoot reached the allocator, which needed 15 GPRs against a budget
// of 14 and gave up.
//
// Such a value wants the cross-block spill instead: one store after its def, one
// reload in each block that uses it. Only the standard per-block path can select
// a victim this way, and it tested for a def in the block without also testing
// for a use.
//
// FLAGS: -O0
// OUTPUT: 123
// EXIT: 0
extern int printf(char* fmt, int x);
double f0(double p0, int p1, double p2, double p3, double p4, double p5, double p6, int p7, int p8, int p9, int p10, double p11) {
    if ((((p8 & 1023) | 801) > (p1 - 12))) {
        for (int i37 = 0; i37 < 3; i37++) {
        }
        if ((p1 < p1)) {
            if ((p1 < p1)) {
                if ((p7 >= p1)) {
                    for (int i37 = 0; i37 < 2; i37++) {
                        if ((p7 >= -1)) {
                            if ((p10 >= p10)) {
                            }
                        }
                    }
                    for (int i32 = 0; i32 < 1; i32++) {
                    }
                }
                if ((p8 >= 58)) {
                    if ((p7 >= p10)) {
                        for (int i59 = 0; i59 < 4; i59++) {
                            if ((i59 == p1)) {
                                if ((-37 >= p10)) {
                                    if ((-62 > p7)) {
                                        if ((-96 == p7)) {
                                            if ((p7 != p8)) {
                                                if ((p10 != p10)) {
                                                    if ((35 == p9)) {
                                                        if ((49 == p9)) {
                                                            if ((p9 >= p1)) {
                                                                if ((p1 >= p10)) {
                                                                    if ((p8 <= -95)) {
                                                                        if ((96 < -19)) {
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            if ((p9 > p1)) {
                                                            }
                                                        }
                                                        if ((57 >= p7)) {
                                                            if ((p10 >= p9)) {
                                                                if ((p10 == p8)) {
                                                                    if ((p8 != p1)) {
                                                                        if ((i59 >= p1)) {
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                if ((p9 <= i59)) {
                                                    if ((p9 >= p8)) {
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for (int i82 = 0; i82 < 4; i82++) {
                            if ((43 > p10)) {
                            }
                        }
                        if ((p7 < 29)) {
                            for (int i78 = 0; i78 < 2; i78++) {
                            }
                            if ((p7 != p9)) {
                                if ((56 > p9)) {
                                }
                            }
                        }
                    }
                    for (int i67 = 0; i67 < 5; i67++) {
                        if ((i67 <= i67)) {
                            if ((p10 >= p1)) {
                            }
                        }
                    }
                }
            }
            if ((16 < p7)) {
            }
        }
    }
    return (12.0 + p3);
}
double f1(double p0, double p1, int p2, double p3, int p4, int p5, double p6, double p7) {
    return ((p6 + p3) + ((-15 - p2) & 63));
}
double f2(double p0, int p1, int p2, double p3, double p4, double p5, int p6, int p7, double p8, int p9) {
    return ((p8 - -40.0) - (-27.0 - p3));
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
    int v0 = 0;
    int v1 = -45;
    int v2 = 48;
    int v3 = -40;
    int v4 = 26;
    int v5 = 20;
    int v6 = -35;
    int v7 = 48;
    int v8 = 36;
    int v9 = 46;
    double d0 = 29.0;
    double d1 = 7.0;
    double d2 = -5.0;
    double d3 = 30.0;
    double d4 = -12.0;
    double d5 = 2.0;
    for (int i45 = 0; i45 < 3; i45++) {
        v6 = (((v5 & 255) << 4) - ((v0 & 1023) ^ 364));
    }
    for (int i56 = 0; i56 < 3; i56++) {
    }
    if ((((((d0 + d1) + d2) + d3) + d4) + d5) != 51.0) { return 3; }
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
