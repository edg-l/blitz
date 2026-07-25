// KNOWN FAILING -- reproducer for a real bug. Do not "fix" by weakening it.
//
//   cc -O0 and cc -O2   -134
//   blitz -O0            25
//   blitz -O1            cannot allocate registers (see ROADMAP P0)
//
// tests/fuzz/gen_c.py seed 6, reduced from 120 lines to 64 by
// tests/fuzz/reduce.py and then re-initialised by hand: the reducer had deleted
// the `arr[i] = ...` stores, and reading them uninitialised is undefined, so the
// comparison against cc would have been meaningless. With every element written
// the divergence stands, and cc -O0 and cc -O2 agree.
//
// The final sum is wrong while every one of its eighteen terms is right on its
// own -- each term printed alone matches cc. blitz's answer, 25, is exactly
// arr[6] + arr[7], the last two terms, which are the only two still in registers
// rather than reloaded from spill slots. So the accumulator arrives at the last
// two adds holding zero.
//
// Ruled out:
//   * A reload from a slot nothing stored. The BLITZ_VERIFY spill-slot check
//     added alongside this file is clean here, and every slot the chain reads
//     has a store.
//   * The address resolution seam. Barrier operands are positional now, and
//     the addresses in this chain are read correctly.
//   * Parallel-copy sequentialization. Fixed separately in the same session and
//     it changes nothing here.
//
// Next: find which slot's *contents* are wrong, i.e. whether two values share a
// slot. Slots are handed out by a single counter, so a collision would have to
// come from the numbering seams -- `pre_spill_slots`, the splitter's per-round
// `first_slot`, and the global allocator's shift, which distinguishes its own
// slots from the splitter's by number range in `compile/mod.rs`. That last one
// is only safe because `run_phase5` never actually allocates a slot today.
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
