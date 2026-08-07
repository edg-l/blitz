// Regression test: coalescing merged block parameters onto values live across the
// whole function, and the merged nodes formed a clique wider than the register
// file.
//
// Every merge here passes the interference test -- a parameter of `f1`'s loop
// header and a constant or incoming parameter that is dead by the time that
// parameter is written genuinely never overlap. But merging replaces two nodes
// with one whose neighbourhood is the union of theirs, and several such merges
// compound: parameters 3, 5 and 9 of one block each picked up a range spanning
// the function, and with the block's parameters already pairwise conflicting the
// result was a 15-clique against 14 allocatable GPRs. No coloring order can
// answer that, so allocation reported a pressure overshoot the splitter had not
// seen -- it measures the graph before coalescing, where the clique does not
// exist.
//
// Declining a merge whose result would have as many significant-degree
// neighbours as there are registers is the Briggs test, and it is what keeps
// coalescing from making a colorable graph uncolorable.
//
// FLAGS: -O1
// OUTPUT: 150
// EXIT: 0
extern int printf(char* fmt, ...);
double f0(double p0, int p1, int p2, int p3, double p4, int p5, int p6, int p7, double p8) {
    return (((49 * p7) & 63) * 29.0);
}
int f1(int p0, int p1, int p2, int p3, int p4, double p5, int p6, int p7, int p8, double p9, int p10) {
    if ((((15 & 1023) & 241) >= 74)) {
    } else {
        for (int i72 = 0; i72 < 2; i72++) {
            if ((p3 <= p6)) {
                if ((49 > -52)) {
                    if ((p10 >= p3)) {
                        if ((79 <= p4)) {
                            if ((71 != i72)) {
                                if ((87 != p6)) {
                                    if ((i72 != -61)) {
                                        if ((p7 == i72)) {
                                            if ((i72 >= 53)) {
                                                if ((-87 <= p1)) {
                                                    if ((-12 <= p10)) {
                                                        if ((p4 >= p4)) {
                                                            if ((p0 < i72)) {
                                                                if ((p2 != p2)) {
                                                                    if ((p7 <= 84)) {
                                                                        if ((-90 <= p4)) {
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            p7 = -47;
                                                        }
                                                        if ((p2 >= p2)) {
                                                            if ((p2 < p1)) {
                                                            }
                                                        }
                                                        if ((p2 >= i72)) {
                                                            if ((30 < p0)) {
                                                                if ((p0 == p6)) {
                                                                    if ((-88 != p1)) {
                                                                        p10 = 30;
                                                                        if ((76 <= 78)) {
                                                                        }
                                                                        p8 = 21;
                                                                        if ((p7 <= i72)) {
                                                                            if ((p8 != i72)) {
                                                                            }
                                                                        }
                                                                        if ((94 >= p1)) {
                                                                            if ((p8 >= p7)) {
                                                                                if ((p4 != 91)) {
                                                                                    p10 = -43;
                                                                                    if ((-59 > p4)) {
                                                                                        p10 = p2;
                                                                                        if ((p1 >= p7)) {
                                                                                            if ((p6 >= p2)) {
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            p2 = p4;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            if ((22 != p3)) {
                                                                if ((89 == p0)) {
                                                                    p4 = 90;
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
        }
    }
    if (((p7 - p3) != -78)) {
        for (int i27 = 0; i27 < 4; i27++) {
            if ((p1 >= p0)) {
                if ((-10 <= 49)) {
                    if ((p8 == p7)) {
                    }
                }
            }
        }
    }
    return ((p6 * p1) & 63);
}
int main() {
    int v0 = 42;
    int v1 = 0;
    int v2 = 8;
    int v3 = -15;
    int v4 = 38;
    int v5 = 38;
    int v6 = 31;
    int v7 = -21;
    int v8 = 40;
    int v9 = 13;
    double d1 = -20.0;
    double d4 = -13.0;
    if ((53 < ((v9 & 1023) | 250))) {
        if (((v8 % 4) == ((54 & 1023) & 889))) {
            if ((v1 >= -83)) {
                for (int i54 = 0; i54 < 2; i54++) {
                    if ((45 < v8)) {
                        if ((v3 > v1)) {
                        }
                    }
                }
            }
            if ((v8 < 26)) {
                for (int i83 = 0; i83 < 3; i83++) {
                }
            }
            if ((-19 > v9)) {
                if ((-92 <= 98)) {
                    if ((v7 < v2)) {
                        if ((v5 > v5)) {
                        }
                    }
                }
            }
        }
    }
    v8 = f1(v7, (-54 + v2), v6, v4, v9, (d1 + 30.0), v8, (v5 + -88), (v0 / 16), d4, (v4 / 8));
    printf("%d\n", (((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9));
}
