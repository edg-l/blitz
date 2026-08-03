// KNOWN FAILING -- reproducer for a real bug. Do not "fix" by weakening it.
//
// Seed 36 of `tests/fuzz/run_fuzz.sh 40 mixed`, reduced. UB-free: `cc -O0`,
// `cc -O2` and `cc -O1 -fsanitize=undefined,address` all print 87.
//
// blitz exits 3 at both -O0 and -O1, which is this program's way of saying its
// double sum came out wrong. BLITZ_VERIFY=strict is clean at both levels, so the
// wrong value sits in a register that was written -- the case the verifier
// explicitly cannot see.
//
// WHAT IS KNOWN, measured rather than read off the disassembly.
//
// Probing this program is delicate: every probe that adds an instruction moves
// register allocation, and the fault moves with it. Adding six equality guards
// made d0 the failing one; adding one guard at a time made d0..d3 pass and d4
// fail. Printing the sum as an int, or calling printf on each double, pushes the
// function past what the allocator can colour, so neither can be measured at all.
//
// The readout that does work replaces this file's final guard with
// `if (<expr> > T) { return 4; }` and bisects T: one compare and one branch, the
// same shape as the guard it replaces, so allocation stays comparable. Both
// optimization levels then agree, which is the sign the probe is not the thing
// being measured.
//
//   d0    cc 678         blitz 678         same
//   d1    cc 88          blitz 88          same
//   d2    cc 24444       blitz 24444       same
//   d3    cc 939         blitz 939         same
//   d4    cc 6233234     blitz 5774234     short by 459000
//   d5    cc 6233449     blitz 5774449     inherits d4's error via d5 += d4 - 40
//
// So one statement is wrong: `d4 = (d4 + (d2 * d5))`. At that point d2 reads
// 24444 and d5 reads 255 under both compilers, and 24444 * 255 = 6233220 is what
// cc uses. blitz's product is 5774220 = 22644 * 255, and 22644 = 36 + 628 * 36
// where 24444 = 36 + 678 * 36 -- so the multiply's d2 operand was computed with
// d0 = 628 rather than 678. The suspect is therefore one statement further back,
// `d2 = (d2 + (d0 * d2))`, reading a d0 that is 50 short.
//
// The IR is correct: `v46 = x86_addsd(v44, v45)` is the updated d0 and
// `v49 = x86_mulsd(v46, v42)` uses it. The schedule and the final assignment are
// self-consistent too (v46 -> XMM3, v42 -> XMM6, v49 -> XMM0). main carries 33
// XMM spill/reload pairs, so the next probe should ask which slot each reload
// reads rather than which register each op names.
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
