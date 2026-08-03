// KNOWN FAILING under one extra interference-graph change -- CORRECT on master
// as it stands. Do not "fix" by weakening it.
//
//   cc -O0, cc -O2, blitz -O0, blitz -O1   0
//   blitz -O1 with the patch below         216
//
// 72 lines, reduced by tests/fuzz/reduce.py from gen_c.py seed 9 shape `args`
// (`f1` extracted with its ten call arguments frozen: all ten agree between
// blitz and cc, only the return value differs, so the fault is inside f1).
// UB-free by the reducer's own check, not by eye.
//
// WHY IT IS HERE. Conservative (Briggs) coalescing is the only thing keeping
// this program correct, and it holds by an accident of degree rather than by an
// interference edge. The `dst` of `Op::TerminatorArgs` is a phantom: the op
// defines nothing and lowering skips it, yet `build_interference_into` gives
// that `dst` an edge to every value live at the terminator, which is every
// argument of the whole parallel copy. Removing those edges -- sound on its own
// terms, and worth 2 seed/level pairs on the `pressure` shape -- lowers every
// terminator argument's degree by one, and Briggs then admits one more merge:
//
//   master (BLITZ_DEBUG=coalesce)     with the patch
//   merge v162 <- v169  deg 17/6      merge v162 <- v169  deg 15/5
//   merge v162 <- v175  deg 23/1      merge v162 <- v175  deg 15/0
//   merge v190 <- v174  deg  7/12     merge v190 <- v174  deg  6/6
//   -- the copy (v174, v175) is       merge v190 <- v162  deg  6/15
//      rejected here --                 <-- this one produces wrong code
//
// The fourth merge unions the pre-loop value with the loop-carried block
// parameter of the same phi chain: v174 is the loop header's parameter
// (ClassId 297), v175 the loop exit's parameter (296), v169 the preheader's
// argument (356), v190 the back edge's argument (305). Every phi copy along
// that chain becomes an identity and is elided, R14 keeps p1's entry value 62
// where p4's 0 belongs, and `((p1 & 255) << 2) & 216` reads 216 instead of 0.
//
// So the interference graph is missing an edge between two values that do
// overlap, and the Briggs degree test was masking it. That is the ROADMAP P0
// statement "the allocator's liveness disagrees with the emitted code's",
// reached by a concrete path. `BLITZ_VERIFY=1` and `=strict` are both silent:
// after the merge the two values are one VReg, so no two live ranges share a
// register and `verify_register_sharing` has nothing to compare.
//
// MEASURED, not argued:
//   - `BLITZ_DEBUG=coalesce` prints the four merges above with their degrees.
//     The pre-merge graph has no v190-v162 edge in either build, so the first
//     interference check passes and Briggs is the only gate.
//   - Suppressing coalescing entirely makes the patched build print 0.
//   - The IR is byte-identical between the two builds (`--emit-ir`), as is the
//     phi copy list; only the VRegs the classes resolve to differ.
//   - `gdb` at the `mov %r14d,%esi` feeding the return: R14 = 62.
//
// THE PATCH, for whoever picks this up:
//
//   Op::has_no_result() -> true for StoreBarrier | VoidCallBarrier |
//   TerminatorArgs(_); `continue` on it in the def-interference loop of
//   regalloc/interference.rs::build_interference_into, and skip counting the
//   def in compile/split.rs::compute_pressure_for_class.
//
// It fixes `pressure` seeds 12 at both levels, is neutral on `mixed` (40/40),
// and regresses only this program. It should land together with the missing
// interference edge, not before it.

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
}
