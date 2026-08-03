// Regression test: a phi copy wrote the register the destination parameter's
// e-class resolved to, while the target block read the register its own
// `BlockParam` instruction carried.
//
// A block parameter can have two VRegs. `build_phi_copies` asked the class, which
// named a VReg the allocator had put in RAX; the target block's schedule read
// RSI. The copy therefore wrote a register nobody looked at, and block 39's
// parameter 0 -- the `i69` loop counter, whose incoming argument is the constant
// 0 -- started at whatever RSI held, which was p1's 62. `62 < 5` is false, so the
// loop was skipped, `p1 = p4` inside it never ran, and
// `((p1 & 255) << 2) & 216` returned 216 instead of 0.
//
// Nothing downstream could see it: RSI was written by the argument setup, so
// def-before-use holds, and no two live ranges shared a register -- the two VRegs
// for the parameter never overlap, one is simply not the one being read.
//
// Reduced by tests/fuzz/reduce.py from gen_c.py seed 9 shape `args`, with `f1`
// extracted and its ten arguments frozen: all ten agreed between blitz and cc, so
// the fault was inside f1. It reproduced only once `Op::TerminatorArgs` stopped
// giving its phantom `dst` interference edges, because that lowered every
// terminator argument's degree by one and let conservative coalescing admit one
// more merge -- which is what put both of the parameter's VRegs in play.
//
// EXIT: 0
// OUTPUT: 0

extern int printf(char* fmt, int x);
int f1(int p0, int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8, int p9) {
    if (((p0 / 16) != (p5 + p3))) {
        if ((p5 != p8)) {
            if ((p6 >= 12)) {
                for (int i24 = 0; i24 < 2; i24++) {
                    if ((-80 < p5)) {
                        if ((p6 <= p6)) {
                        }
                        if ((p1 < 33)) {
                            if ((-15 == -20)) {
                            }
                        }
                    }
                }
            } else {
                if ((29 < p1)) {
                    p2 = p9;
                }
                if ((67 >= p7)) {
                    for (int i68 = 0; i68 < 3; i68++) {
                    }
                    for (int i90 = 0; i90 < 1; i90++) {
                    }
                }
            }
        }
        for (int i69 = 0; i69 < 5; i69++) {
            if ((49 != -53)) {
                if ((p5 > 96)) {
                    if ((p6 <= p6)) {
                        if ((-2 != p5)) {
                        }
                        if ((p9 != p9)) {
                            if ((p7 == 49)) {
                                p5 = p8;
                            }
                        }
                    }
                    p9 = p3;
                } else {
                    if ((p9 <= p7)) {
                        p7 = -69;
                    } else {
                        if ((p6 <= p9)) {
                        }
                        p7 = -43;
                    }
                }
                if ((p0 <= i69)) {
                    if ((i69 == p6)) {
                        if ((-26 <= -42)) {
                        }
                    }
                }
                if ((p3 >= 54)) {
                }
            }
            p1 = p4;
        }
        if ((p4 <= -96)) {
            for (int i37 = 0; i37 < 5; i37++) {
                if ((p5 != p6)) {
                }
            }
        }
    }
    return ((((p1 & 255) << 2) & 1023) & 216);
}
int main() {
    printf("%d\n", f1(0, 62, -60, -84, 0, 192, -1633, -13, 0, 264));
    return 0;
}
