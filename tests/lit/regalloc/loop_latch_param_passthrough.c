// Regression test: a loop latch handing every block param straight back to its
// header, while the header spills those params to slots.
//
// blitz printed 25 where cc prints -134 at both -O0 and -O2. Perturbing one term
// of the sum at a time by +1000, and requiring cc's answer to move by exactly
// +1000 so the probe is known linear, showed all eight `arr[i]` terms
// contributing correctly and all ten `vN` terms contributing NOTHING: the
// integer half of the sum was the constant -3, unrelated to its inputs. Every
// `vN` is a compile-time constant that reaches the sum as a block param of b4.
//
// Two seams disagreed about where those params live at the latch's exit, and
// each was wrong on its own.
//
// Liveness (`compute_phi_uses`) resolved the arg class at the latch's exit,
// which nothing covers once the header spills the param: the splitter truncates
// the original segment at its SpillStore, and only the reloads it inserts
// elsewhere have segments. The class resolved to no VReg at all, so the value
// looked dead over the loop body, and the allocator gave RCX -- block param 15's
// register -- to the latch's own loop-counter increment. The header then
// re-spilled the clobbered register on the second iteration.
//
// Emission (`build_phi_copies`) resolved the same class through the
// coalesce-alias fallback in the per-block map. That fallback was built by
// calling `insert_single` once per segment, which replaces the whole class, so
// the surviving answer was whichever segment came last: a reload from an
// unrelated block, sharing one scratch register with every other reload. Ten
// phi copies all read RAX.
//
// Fixing the first exposed a third bug: the copies it then produced included a
// swap through R11, which was both the hard-coded parallel-copy scratch and an
// allocatable register, and `sequentialize_copies` spun forever trying to park
// R11 in itself.
//
// tests/fuzz/gen_c.py seed 6, reduced from 120 lines to 64 by
// tests/fuzz/reduce.py and then re-initialised by hand: the reducer had deleted
// the `arr[i] = ...` stores, and reading them uninitialised is undefined, so the
// comparison against cc would have been meaningless.
//
// Note that several `vN` share a value (-42 appears twice), so they share an
// e-class and legitimately resolve to one VReg. Two phi copies reading the same
// source register is expected here, not a bug.
//
// Pinned to -O0: blitz -O1 still cannot allocate registers for this program
// (ROADMAP P0), which is a separate failure from the one under test here.
//
// FLAGS: -O0
// OUTPUT: -134
// EXIT: 0
//
extern int printf(char* fmt, int x);
int f0(double p0, double p1, double p2, double p3, int p4, int p5, int p6) {
    for (int i9 = 0; i9 < 1; i9++) {
    }
}
double f1(double p0, int p1, double p2, int p3, int p4, double p5, double p6, int p7, int p8, double p9, double p10) {
    if (((46 - p1) >= ((3 & 255) >> 1))) {
        for (int i51 = 0; i51 < 3; i51++) {
        }
        if ((88 < p1)) {
            for (int i82 = 0; i82 < 1; i82++) {
                if ((p3 < i82)) {
                    if ((i82 < i82)) {
                        if ((p7 <= p1)) {
                        }
                    }
                }
            }
        }
    }
    return ((p6 - p6) + p0);
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
    int v0 = -13;
    int v1 = -15;
    int v2 = -42;
    int v3 = 5;
    int v4 = -42;
    int v5 = -43;
    int v6 = 39;
    int v7 = 6;
    int v8 = -40;
    int v9 = -17;
    double d0 = 24.0;
    double d1 = 26.0;
    double d2 = 12.0;
    double d3 = -29.0;
    double d4 = -14.0;
    double d5 = -6.0;
    if (((((v4 & 1023) ^ 207) - (v7 / 8)) < ((v0 / 16) / 8))) {
    }
    d5 = f1((d0 + -38.0), v2, (d4 * d1), (v5 * v8), ((-87 & 255) << 1), d1, (d0 * d1), v1, (-47 / -3), (d5 - d2), d5);
    d0 = (d0 + (d1 - d5));
    d1 = (d1 + (d0 + d0));
    d2 = (d2 + d4);
    d3 = (d3 + (d0 * d4));
    d4 = (d4 + d1);
    d5 = (d5 + (-12.0 * d5));
    for (int i93 = 0; i93 < 1; i93++) {
    }
    d1 = f1(d3, ((v1 & 1023) & 992), (d2 * d2), ((v7 & 1023) ^ 131), (v7 / 2), d4, d1, (v1 % 8), v1, d3, (d5 + d2));
    d0 = (d0 + (2.0 * 21.0));
    d1 = (d1 + (d0 * d1));
    d2 = (d2 + (d5 - -5.0));
    d3 = (d3 + (d2 + -37.0));
    d4 = (d4 + (10.0 * d3));
    d5 = (d5 + d4);
    if ((((((d0 + d1) + d2) + d3) + d4) + d5) != -115183.0) { return 3; }
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
