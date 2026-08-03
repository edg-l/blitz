// KNOWN FAILING -- wrong value at BOTH levels. Do not "fix" by weakening it.
//
//   cc -O0 and cc -O2   666
//   blitz -O0            742
//   blitz -O1            742
//
// Wrong at both levels is a new signature here: every bug of the last three
// sessions was wrong at one level and right or uncompilable at the other, so the
// -O0-vs-O1 oracle carried them. This one only the reference compiler sees.
//
// ONE WRONG TERM, and it is a store that should not have happened. `arr[4]` reads
// 81 where its initialiser 5 belongs, which is the whole +76; the reference never
// writes that element. Every other term of the sum agrees, including arr[6] at 608
// and arr[7] at 38, so some store's index came out 4.
//
// Read off the unmodified binary with `read_frame.py --sum-chain` against a `cc`
// build whose printf is split into one call per term. Perturbation was not needed
// and would have been the wrong tool: this program allocates comfortably, so start
// from the store, not from the register file.
//
// 247 lines, reduced from gen_c.py seed 58 shape mixed (666 lines) with all eight
// `arr[k] =` initialisers restored afterwards -- reduce.py deleted three and its
// UB guard cannot see it, because the loops write arr through computed indices.
// The unreduced program is wrong by +8 rather than +76, so the reduction found a
// larger instance of the same shape, not a different bug: still one array element.
//
// EXIT: 0
// OUTPUT: 666
extern int printf(char* fmt, int x);
double f0(double p0, int p1, double p2, int p3, int p4, double p5, int p6, double p7, double p8) {
    for (int i17 = 0; i17 < 4; i17++) {
        if ((p6 <= -83)) {
            if ((p1 <= -84)) {
            }
            if ((p1 == p6)) {
                if ((p3 != p1)) {
                }
            }
        }
        if ((p1 >= p1)) {
            if ((p4 < p6)) {
                if ((-35 > p1)) {
                    if ((56 == p4)) {
                        if ((64 <= 65)) {
                            if ((-68 < 84)) {
                                if ((p1 >= 65)) {
                                    if ((31 <= 31)) {
                                    }
                                }
                            }
                        }
                    }
                    if ((p3 < 0)) {
                        if ((5 == p4)) {
                            if ((90 == p6)) {
                            }
                        }
                        if ((p6 >= 1)) {
                        }
                    }
                }
            }
        }
        if ((-23 > p3)) {
            if ((p1 > 87)) {
            }
        }
    }
    return (p0 - (p7 * p7));
}
int f1(double p0, int p1, int p2, int p3, double p4, double p5, double p6, int p7, double p8) {
    if (((p7 / -3) == (89 + -71))) {
        for (int i31 = 0; i31 < 1; i31++) {
            if ((p3 != p1)) {
            }
        }
        for (int i15 = 0; i15 < 4; i15++) {
        }
        if ((p1 != -23)) {
            if ((p7 >= p7)) {
                for (int i87 = 0; i87 < 4; i87++) {
                    if ((i87 != -62)) {
                        if ((-68 >= -4)) {
                            if ((p7 > p3)) {
                                if ((i87 < -16)) {
                                    if ((p2 <= p3)) {
                                        if ((i87 >= -98)) {
                                            if ((p7 != 51)) {
                                                if ((p7 != p7)) {
                                                }
                                            }
                                            if ((p2 != 55)) {
                                            }
                                        }
                                    }
                                }
                            }
                            if ((p7 >= p1)) {
                                if ((-15 != -14)) {
                                    if ((i87 < p1)) {
                                        if ((20 <= p7)) {
                                            if ((-7 < 62)) {
                                                if ((i87 >= -51)) {
                                                    if ((p1 > p2)) {
                                                        if ((-42 <= -81)) {
                                                        }
                                                    }
                                                }
                                                if ((p1 > -85)) {
                                                    if ((p2 > 19)) {
                                                        if ((i87 >= p7)) {
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if ((-60 >= p1)) {
                                    if ((i87 >= p3)) {
                                        if ((i87 == -69)) {
                                            if ((-17 >= p3)) {
                                                if ((p7 >= 7)) {
                                                    if ((p2 <= i87)) {
                                                        if ((p2 != p3)) {
                                                        }
                                                    }
                                                }
                                            }
                                            if ((i87 < p3)) {
                                                if ((i87 != p7)) {
                                                    if ((i87 != -23)) {
                                                        if ((p2 != p2)) {
                                                            if ((p3 != i87)) {
                                                                if ((94 > -48)) {
                                                                    if ((35 > p1)) {
                                                                        if ((-54 < p1)) {
                                                                            if ((p3 > p3)) {
                                                                            }
                                                                        }
                                                                        if ((-68 > p7)) {
                                                                            if ((p7 > i87)) {
                                                                            }
                                                                        }
                                                                        if ((-24 >= p7)) {
                                                                            if ((-98 >= i87)) {
                                                                                if ((p1 == i87)) {
                                                                                    if ((p2 != -33)) {
                                                                                        if ((91 != 92)) {
                                                                                            if ((p3 <= p3)) {
                                                                                                if ((p7 >= i87)) {
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                    if ((p2 != p7)) {
                                                                        if ((89 == 30)) {
                                                                        }
                                                                    }
                                                                }
                                                                if ((p2 >= 17)) {
                                                                    if ((94 == p2)) {
                                                                        if ((57 < 59)) {
                                                                            if ((p1 >= p7)) {
                                                                            }
                                                                            if ((i87 > i87)) {
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
                                        if ((p7 > p7)) {
                                        }
                                    }
                                }
                                if ((p7 <= p3)) {
                                    if ((p3 < i87)) {
                                        if ((32 > -33)) {
                                            if ((p7 == p3)) {
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if ((p1 != p2)) {
                        }
                    }
                }
                if ((p3 != 97)) {
                    for (int i40 = 0; i40 < 5; i40++) {
                    }
                    if ((p2 < -24)) {
                        if ((p7 != 36)) {
                            if ((p3 <= p7)) {
                            }
                        }
                    }
                }
            }
            for (int i48 = 0; i48 < 4; i48++) {
                if ((-15 == i48)) {
                }
            }
        }
    }
    for (int i29 = 0; i29 < 2; i29++) {
        if ((-38 >= p3)) {
            if ((p2 >= i29)) {
            }
            if ((p1 != p3)) {
                if ((p7 == p7)) {
                }
            }
        }
    }
    return -7;
}
double f2(int p0, double p1, double p2, int p3, double p4, int p5, int p6, int p7, double p8, int p9, int p10) {
    return p2;
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
    int v0 = 2;
    int v1 = -11;
    int v2 = 14;
    int v3 = 38;
    int v4 = 46;
    int v5 = -11;
    int v6 = -2;
    int v7 = -26;
    int v8 = 10;
    int v9 = -50;
    double d0 = -25.0;
    double d1 = -21.0;
    double d2 = -5.0;
    double d3 = -16.0;
    double d4 = 13.0;
    double d5 = -3.0;
    arr[((v1 % 5)) & 7] = v3;
    for (int i16 = 0; i16 < 6; i16++) {
        arr[(v3) & 7] = ((v3 & 255) << 4);
    }
    for (int i78 = 0; i78 < 1; i78++) {
        if ((((-64 & 1023) ^ 687) > v2)) {
            if ((v5 == v2)) {
            }
        }
        if ((((i78 & 255) >> 2) == (v2 - v6))) {
        } else {
            if ((v2 < v6)) {
                arr[((58 - v2)) & 7] = 81;
            }
            arr[(((-64 & 255) << 4)) & 7] = (i78 * v0);
        }
    }
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
