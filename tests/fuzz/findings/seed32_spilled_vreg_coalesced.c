// KNOWN FAILING -- reproducer for a real bug. Do not "fix" by weakening it.
//
//   cc -O0 and cc -O2   83
//   blitz -O1           179
//   blitz -O0           cannot allocate registers (ROADMAP P0)
//
// 44 lines. Reduced from gen_c.py seed 32 with reduce.py, then re-initialised by
// hand TWICE: the reducer deleted the `arr[i] = ...` stores both times, and
// reading them uninitialised is undefined, which made the first "divergence"
// meaningless. Always re-check a reduction for uninitialised reads.
//
// WHAT IS WRONG, by perturbing one term at a time by +1000 and requiring cc's
// answer to move by exactly +1000:
//
//   * All eight arr[i] terms and eight of the ten vN terms are correct.
//   * v5 and v8 are wrong. Both hold the constant 1, so they share an e-class,
//     and the total is 96 too high -- 48 per term.
//
// THE MECHANISM, established from the asm and the allocator's own dumps:
//
// The class of `1` is VReg 10 in R14, which `mov r14d,0x1` writes once and
// nothing clobbers. Its phi copies are consistent: R14 -> R14 for one param and
// R14 -> RSI (VReg 49) for the other. The corruption is in a spill store:
//
//   1b5: mov esi,r14d          ; RSI = 1, the phi copy
//   1cd: mov [rsp+0xf0],rsi    ; SpillStore(30) of VReg 49, correct
//   264: mov esi,0xfffffffa    ; RSI reused as an idiv divisor
//   2ae: mov rsi,[rsp+0xd8]    ; and again
//   2f6: mov [rsp+0xb0],rsi    ; SpillStore(22) of VReg 112 -- stores THAT
//
// Slot 22 (rsp+0xb0) is read by the printf sum, so the wrong value is what the
// term contributes. The store's operand is VReg 112, and **VReg 112 has no
// register in the final assignment at all**: coalescing merged it onto the VReg
// holding RSI. That VReg is 49, whose segment the splitter TRUNCATED at
// {block 2, inst 7} when it cross-block-spilled it (`insert b2 before [7]:
// SpillStore(30)([49])`, reloads registered at {2,MAX}, {3,1}, {14,3}).
//
// So coalescing merged a VReg whose live range has a hole in it -- the hole the
// spill created -- with a later VReg, and the allocator filled the hole with the
// idiv divisor. The merged value is only valid in part of its own range.
//
// This is the landmine ROADMAP P0 records: the coalesce-alias step collapses a
// class to one VReg and discards the ranges. The next step is NOT to guess at
// coalescing: build the missing machine-verifier check first -- no two
// overlapping live ranges may share a physical register -- which names this
// mechanically and covers the other programs the splitter change made visible.
//
// FLAGS: -O1
// OUTPUT: 83
// EXIT: 0
extern int printf(char* fmt, int x);
double f0(int p0, int p1, double p2, int p3, double p4, double p5, double p6, int p7, double p8, double p9, int p10, int p11) {
    return -21.0;
}
int f1(double p0, int p1, double p2, int p3, int p4, double p5, double p6, double p7) {
    return 93;
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
    int v1 = 20;
    int v2 = -29;
    int v3 = 12;
    int v4 = 27;
    int v5 = 1;
    int v6 = 7;
    int v7 = 33;
    int v8 = 1;
    int v9 = -19;
    double d0 = 27.0;
    double d1 = -18.0;
    double d2 = -3.0;
    double d3 = 9.0;
    double d4 = -3.0;
    double d5 = 19.0;
    if (((((v9 * v6) & 1023) ^ 807) < v5)) {
    }
    d0 = f0(64, -60, -4.0, (v3 % 3), (d2 * -36.0), (((-11 & 255) >> 2) & 63), d5, (-7 - 55), d1, -37.0, ((v6 & 1023) ^ 313), v8);
    if (((((v5 % -6) & 1023) | 96) == (v6 * v3))) {
        if (((v5 % 2) == ((34 & 255) << 1))) {
        }
    }
    for (int i21 = 0; i21 < 5; i21++) {
    }
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
