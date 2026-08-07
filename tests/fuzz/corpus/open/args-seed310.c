// The -O1 register allocator gets one term of this sum wrong: 79 where 82
// belongs.
//
// Nothing in the optimization set is responsible. Every pass can be turned off
// one at a time, or all together, and -O1 still prints 79; forcing the -O0
// allocator on at -O1 with `BLITZ_PASSES=+fast-regalloc` prints 82. What is
// left is the colouring allocator, the splitter in front of it, and coalescing.
//
// The -O0-vs-O1 oracle could not see this while both levels shared an
// allocator. It is the first bug the second implementation has found.
//
// 237 lines, reduced from gen_c.py seed 310 shape args. `arr[0]` and `arr[6]`
// have no initializer left but both are written by the loops before the sum
// reads them -- checked by hand, since reduce.py's guard cannot see it.
//
// OUTPUT: 82
extern int printf(char* fmt, int x);
int f0(int p0, int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8, int p9, int p10, int p11) {
    p11 = (((p1 + p3) & 255) << 3);
    p3 = ((p8 & 1023) ^ 229);
    if ((((p10 & 1023) | 973) <= (p0 + -79))) {
        for (int i39 = 0; i39 < 3; i39++) {
        }
        if ((p11 != p4)) {
            if ((p8 <= -65)) {
            }
            if ((p6 <= p4)) {
                for (int i91 = 0; i91 < 4; i91++) {
                }
                if ((-63 < -63)) {
                    for (int i49 = 0; i49 < 2; i49++) {
                    }
                    for (int i71 = 0; i71 < 2; i71++) {
                        if ((p2 > -26)) {
                        }
                    }
                }
            }
        }
    } else {
        if ((p11 <= p7)) {
            if ((57 <= 6)) {
                if ((p7 != -21)) {
                    for (int i11 = 0; i11 < 6; i11++) {
                        if ((p1 == p11)) {
                            if ((9 == p6)) {
                                if ((i11 != p10)) {
                                    if ((p10 <= p7)) {
                                        if ((-38 >= 32)) {
                                            if ((i11 >= p10)) {
                                                if ((p1 < -40)) {
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if ((p9 <= -33)) {
                        if ((p2 != p2)) {
                            for (int i84 = 0; i84 < 2; i84++) {
                            }
                        }
                    }
                }
            }
        } else {
            if ((-33 != p6)) {
                p4 = 50;
                if ((-57 == p9)) {
                    for (int i6 = 0; i6 < 4; i6++) {
                        if ((p7 != p1)) {
                        }
                        if ((p6 > 99)) {
                            if ((p6 > 85)) {
                                if ((p2 >= p6)) {
                                    if ((p3 <= p0)) {
                                        if ((p10 > p3)) {
                                            if ((p4 < p8)) {
                                                if ((4 == p1)) {
                                                }
                                                if ((-36 < p10)) {
                                                    if ((-29 != p0)) {
                                                        if ((p8 <= p1)) {
                                                        }
                                                    }
                                                }
                                            }
                                            if ((p9 == -72)) {
                                                if ((-42 < p3)) {
                                                    if ((p6 < 39)) {
                                                        if ((p2 > p10)) {
                                                            if ((-76 <= p4)) {
                                                                if ((-49 < i6)) {
                                                                }
                                                                if ((p4 == 33)) {
                                                                    if ((-2 < -47)) {
                                                                    }
                                                                }
                                                            }
                                                            if ((p5 < p10)) {
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            if ((i6 > 41)) {
                                                if ((85 >= p6)) {
                                                    if ((18 > -86)) {
                                                        if ((p0 == -65)) {
                                                            if ((p6 > p3)) {
                                                                if ((-42 > p2)) {
                                                                    if ((i6 != p8)) {
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    if ((p2 != p11)) {
                                                        if ((p4 != p7)) {
                                                            if ((p10 == p3)) {
                                                                if ((90 == p7)) {
                                                                    if ((p7 <= p3)) {
                                                                        if ((i6 <= p8)) {
                                                                            if ((p11 > i6)) {
                                                                                if ((i6 != p2)) {
                                                                                    if ((p3 < p3)) {
                                                                                        if ((p2 == p6)) {
                                                                                            if ((53 <= 21)) {
                                                                                                if ((p0 != p7)) {
                                                                                                    if ((-99 > p5)) {
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                        if ((i6 <= 61)) {
                                                                                            if ((-85 <= p11)) {
                                                                                                if ((p0 > p2)) {
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            if ((p10 > p6)) {
                                                                            }
                                                                        }
                                                                        if ((p9 > -2)) {
                                                                            if ((i6 <= p4)) {
                                                                                if ((72 == i6)) {
                                                                                    if ((-96 <= p8)) {
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                                if ((p4 == p1)) {
                                                                    if ((-27 <= p1)) {
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
                    if ((p5 <= p7)) {
                        for (int i96 = 0; i96 < 5; i96++) {
                        }
                    }
                }
            }
        }
    }
    return (((p4 & 255) >> 3) - -6);
}
int f1(int p0, int p1, int p2, int p3, int p4, int p5, int p6, int p7) {
    if ((p1 < ((p0 & 1023) & 1022))) {
        for (int i97 = 0; i97 < 6; i97++) {
            if ((p0 >= 59)) {
            }
        }
        for (int i87 = 0; i87 < 3; i87++) {
            if ((p4 <= p3)) {
                if ((-91 == p6)) {
                }
            }
        }
    }
    return 80;
}
int main() {
    int arr[8];
    arr[1] = -4;
    arr[2] = -1;
    arr[3] = 2;
    arr[4] = 5;
    arr[5] = 8;
    arr[7] = 14;
    int v0 = -39;
    int v1 = -5;
    int v2 = 19;
    int v3 = -50;
    int v4 = 17;
    int v5 = 39;
    int v6 = 42;
    int v7 = -42;
    int v8 = -34;
    int v9 = -38;
    if ((((v4 & 1023) | 635) != ((v2 & 255) << 0))) {
        if ((v0 < (-87 / 16))) {
            if ((v6 > v3)) {
                if ((v8 >= v3)) {
                    if ((v3 == v0)) {
                        for (int i89 = 0; i89 < 3; i89++) {
                        }
                    }
                    if ((v4 >= v6)) {
                        for (int i88 = 0; i88 < 3; i88++) {
                            if ((-55 <= 36)) {
                            }
                        }
                    }
                }
                for (int i0 = 0; i0 < 3; i0++) {
                    if ((v8 <= v8)) {
                    }
                    arr[((v6 - v9)) & 7] = v8;
                }
            }
            for (int i82 = 0; i82 < 6; i82++) {
                if ((i82 <= 26)) {
                }
            }
        }
        for (int i97 = 0; i97 < 6; i97++) {
            v4 = ((16 & 255) << 4);
            if ((v6 <= v0)) {
            } else {
                arr[((v1 / 2)) & 7] = -26;
            }
        }
    }
    v6 = f0(v8, (v4 - v7), ((v9 & 1023) & 558), ((-37 & 255) >> 1), ((-35 & 255) >> 3), ((v8 & 255) << 1), ((v2 & 255) << 2), ((-36 & 1023) & 453), ((v7 & 255) >> 3), ((v9 & 1023) | 803), (-62 % 5), ((v8 & 1023) & 760));
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
