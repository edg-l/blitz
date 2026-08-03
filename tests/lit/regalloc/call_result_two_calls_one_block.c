// Regression test: a call's result was copied out of the ABI return register
// only when the result CLASS resolved to a register, so with two calls in one
// block the copy was skipped and nothing wrote the register the consumers read.
//
// `main` calls f1 twice. SysV returns a double in XMM0, and lowering emits a copy
// from XMM0 into the register the result was allocated -- but it asked which
// register by resolving the result's e-class, and a class can name a different
// VReg than the one the schedule carries. Here it named one whose register was
// XMM0 already, so no copy was emitted, while the consumer read XMM1. The spill
// of `d0` immediately after the call therefore stored XMM1, a leftover argument,
// and `d0 = f1(...)` became -14 instead of 36.
//
// Nothing downstream can see this. The register is written -- by the argument
// setup -- so def-before-use holds, and no two values share a register or a slot.
// The barrier instruction's own dst is the answer, as it is for terminator
// arguments and for a division's projections.
//
// Related: `add_call_precolors_for_block` pins a call result to the ABI return
// register only when its block holds exactly one call, which is why one call per
// block never showed this. Widening that guard is not the fix -- two results
// pinned to one register collide whenever their ranges overlap.
//
// How the fault was located, since the program resists probing: every added
// instruction moves register allocation and the fault with it, so the value was
// read out by bisecting a same-size comparison (tests/fuzz/read_double_sum.py),
// which said d4 was short by exactly 459000, and then gdb on the unmodified
// binary walked the operands back to the store after the call.
//
// EXIT: 0
// OUTPUT: 87

extern int printf(char* fmt, int x);
int f0(int p0, int p1, double p2, double p3, int p4, int p5, double p6, int p7, int p8) {
    if ((((p4 & 1023) ^ 287) == p1)) {
        if ((p1 <= p8)) {
            for (int i1 = 0; i1 < 1; i1++) {
                if ((p1 == -38)) {
                    if ((69 == p8)) {
                        if ((p8 == p1)) {
                            if ((p8 <= i1)) {
                                if ((p8 <= i1)) {
                                }
                            }
                        }
                    }
                }
            }
            for (int i73 = 0; i73 < 5; i73++) {
                if ((p7 <= 17)) {
                    if ((-11 <= -38)) {
                        if ((p4 < p7)) {
                            if ((p5 < p0)) {
                                if ((p5 >= p7)) {
                                }
                            }
                        }
                        if ((p1 != p5)) {
                            if ((-15 >= p1)) {
                            }
                            if ((43 > -69)) {
                                if ((-41 <= p0)) {
                                    if ((59 == p5)) {
                                        if ((p7 != p4)) {
                                            if ((-30 >= p4)) {
                                                if ((p7 >= p5)) {
                                                    if ((-54 != p4)) {
                                                        if ((p8 != 45)) {
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 3;
}
double f1(int p0, double p1, double p2, int p3, int p4, double p5, double p6, double p7, double p8, int p9, int p10) {
    if ((p4 == ((p3 & 1023) | 389))) {
        for (int i58 = 0; i58 < 3; i58++) {
            if ((-77 != 68)) {
                if ((p10 > p4)) {
                }
            }
        }
        if ((p10 < -60)) {
            if ((79 > p9)) {
                if ((p4 == p10)) {
                    for (int i92 = 0; i92 < 3; i92++) {
                    }
                    if ((p0 <= p10)) {
                        for (int i90 = 0; i90 < 4; i90++) {
                            if ((95 == p0)) {
                            }
                        }
                        if ((64 != p4)) {
                            if ((p0 > 46)) {
                            }
                        }
                    }
                }
                for (int i98 = 0; i98 < 6; i98++) {
                    if ((p9 != -97)) {
                        if ((p10 != -77)) {
                            if ((50 > p10)) {
                                if ((p10 <= p3)) {
                                    if ((-47 <= i98)) {
                                    }
                                    if ((p4 <= -79)) {
                                        if ((i98 == p9)) {
                                            if ((p0 != p0)) {
                                                if ((p3 == p9)) {
                                                    if ((p3 > p9)) {
                                                        if ((-14 != 55)) {
                                                        }
                                                    }
                                                    if ((p3 >= i98)) {
                                                    }
                                                }
                                            }
                                            if ((i98 == p10)) {
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return (((-84 & 1023) & 118) & 63);
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
    int v0 = 40;
    int v1 = -23;
    int v2 = 21;
    int v3 = -28;
    int v4 = 44;
    int v5 = 44;
    int v6 = -35;
    int v7 = 8;
    int v8 = -48;
    int v9 = 36;
    double d0 = 13.0;
    double d1 = 18.0;
    double d2 = 15.0;
    double d3 = 23.0;
    double d4 = 14.0;
    double d5 = 17.0;
    d0 = f1(v1, (12.0 + d3), (d4 - 28.0), v4, (13 + v7), (d4 * d0), (d2 + -24.0), -24.0, 17.0, ((v9 & 1023) ^ 832), (-60 % 8));
    d0 = (d0 + (d3 * 15.0));
    d1 = (d1 + ((-38 * v4) & 63));
    d3 = (d3 + (d4 * d5));
    d5 = (d5 + (d4 * d5));
    d2 = f1(((-28 & 1023) | 866), (37.0 * 21.0), (((v2 & 1023) ^ 122) & 63), (v7 % 4), ((v4 & 255) << 0), d3, d5, (d4 + d1), (d5 - d5), ((80 & 255) >> 3), ((52 & 255) << 1));
    d0 = (d0 + (d2 + d3));
    d1 = (d1 + d4);
    d2 = (d2 + (d0 * d2));
    d3 = (d3 + d0);
    d4 = (d4 + (d2 * d5));
    d5 = (d5 + (d4 - 40.0));
    if ((((((d0 + d1) + d2) + d3) + d4) + d5) != 12492832.0) { return 3; }
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
