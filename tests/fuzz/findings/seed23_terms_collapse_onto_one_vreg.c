// KNOWN FAILING -- reproducer for a real bug. Do not "fix" by weakening it.
//
//   cc -O0 and cc -O2   -4
//   blitz -O0           15
//   blitz -O1           -21
//
// 49 lines, reduced from gen_c.py seed 23. UB-free: verified by the reducer's
// own check (`cc -O2 -Werror=return-type -Werror=uninitialized
// -Werror=maybe-uninitialized`) rather than by eye. The first reduction of this
// program was NOT UB-free -- it had lost both `return`s and seven of the eight
// `arr[i] =` initialisers -- which is what prompted that check.
//
// The program compiles at all only since dd8449c (degree-order coloring retry);
// before that both levels failed to allocate. So this is a pre-existing
// miscompile that the allocation failure was hiding, not a regression: the
// whole-corpus check at that commit found zero seed/level pairs going from
// correct to wrong.
//
// WHAT IS WRONG, by perturbing one initialiser at a time by +1000 and keeping
// only the probes where cc's answer moves by exactly +1000:
//
//   term      cc delta   blitz delta
//   arr[1]      1000        1025
//   arr[5]      1000        1013
//   v0          1000         -15
//   v1          1000         146
//   v2          1000           0
//   v3          1000        2000
//   v5          1000           0
//   v6          1000          25
//   v7          1000        3000
//   v8          1000           0
//
// v3 is counted twice, v7 three times, and v2/v5/v8 not at all. Several values
// are collapsing onto the same storage. Note the whole double chain d0..d5 is
// dead -- nothing reads it -- so its only role is register pressure, and the
// `for` loop and the `if` are both empty for the same reason.
//
// RULED OUT, each by a check rather than an argument:
//
//   * Register sharing. `BLITZ_VERIFY=1` is silent, so no two VRegs that are
//     live at the same point share a register in the emitted schedules.
//   * Pre-coloring conflicts. Silent too (coloring.rs check_precolorings).
//   * Spill slot collision. Every SpillStore slot in main has exactly one
//     writing VReg; there is no slot written by two different values.
//   * The load/spill sequence for the eight arr[i] terms. It reads correctly:
//     `mov r12,[rsp+0x30]; mov r9d,[r12]; mov [rsp],r9` reloads the address,
//     loads through it into R9, and spills R9 -- and the allocator does assign
//     that LoadResult R9. (Read this one with FULL context: grepping the asm
//     for `mov.*rsp` drops the middle instruction and makes the store look like
//     it reads a register the load never wrote.)
//   * The shape of the sum. All 18 terms are present in the emitted chain,
//     eight from spill slots and two from registers.
//
// MECHANISM, now PROVEN rather than suspected. The four faulty vN terms are
// block params of block 7 that COALESCING MERGED ONTO A VREG WHOSE REGISTER IS
// ALREADY STALE. Tracing the alias chain and the final assignment:
//
//   v103 -> v43 -> R13      v104 -> v3  -> R10
//   v105 -> v6  -> RDI      v106 -> v7  -> RDX
//
// Take v103, the param carrying v0. It coalesces onto v43 = Iconst(7), which
// the allocator put in R13. But v43 has exactly ONE use in the whole schedule,
// `SpillStore(15)([43])`: the splitter slot-spilled it, so from that store on
// the value lives in slot 15 and R13 is free. The allocator reuses R13 twice
// (`mov r13,[rsp+0x30]`, `mov r13d,r12d`) and the sum then reads it:
//
//   1a0: mov r13d,0x7          ; v43 = 7
//   ...: mov [rsp+..],r13      ; SpillStore(15) -- value now lives in the slot
//   206: mov r13,[rsp+0x30]    ; R13 reused, legally: v43 is dead
//   20b: mov r13d,r12d         ; reused again
//   413: mov r14d,r13d         ; the sum reads R13 as v0. Stale.
//
// Nothing writes R13 with v0's value before the sum, because the phi copy that
// would have done so was elided: source and destination coalesced to the same
// VReg, so it looked redundant. It was not -- the source's REGISTER no longer
// holds the value, only its slot does.
//
// This is the P0 landmine in ROADMAP.md: the coalesce-alias step collapses a
// class to one VReg and discards the ranges, so a class whose value has moved
// to a slot still resolves to the register it used to occupy. Note it needs no
// overlapping live ranges, which is why `verify_register_sharing` is silent:
// the fault is a consumer naming a register that is dead, not two live values
// sharing one.
//
// RULED OUT as the fix: excluding every slot-spilled VReg from coalescing
// (filter `copy_pairs` on the operands of SpillStore/XmmSpillStore before
// `allocate_global`). It moves this program from 15 to -3 -- most of the
// corruption gone, one term still wrong -- but over the 40-seed corpus it is a
// net LOSS: five seed/level pairs go from correct to wrong against two fixed,
// because dropping that much coalescing reshuffles allocation everywhere. Too
// blunt. A narrower rule would have to spare a phi source only where the copy
// is emitted before the SpillStore, and the real fix is the P0 refactor:
// build the class-to-VReg map for lowering ONCE from the final post-allocation
// schedules instead of patching a pre-split snapshot three times.
//
// FLAGS: -O0
// OUTPUT: -4
// EXIT: 0
extern int printf(char* fmt, int x);
int f0(double p0, int p1, double p2, double p3, double p4, int p5, int p6, double p7) {
    return 99;
}
int f1(int p0, double p1, int p2, int p3, double p4, int p5, int p6, int p7, int p8, int p9, int p10, int p11) {
    return p11;
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
    int v0 = 7;
    int v1 = 8;
    int v2 = 21;
    int v3 = -10;
    int v4 = -24;
    int v5 = -15;
    int v6 = -4;
    int v7 = 26;
    int v8 = 23;
    int v9 = -34;
    double d0 = -21.0;
    double d1 = 0.0;
    double d2 = 18.0;
    double d3 = 27.0;
    double d4 = 8.0;
    double d5 = -25.0;
    for (int i32 = 0; i32 < 5; i32++) {
    }
    d0 = (d0 + (d4 + 2.0));
    d3 = (d3 + (d3 * d5));
    d4 = (d4 + (d3 + d3));
    d5 = (d5 + (d5 - d1));
    if (((((v1 & 255) << 2) / 7) >= ((v3 + v7) - (v3 / 2)))) {
    }
    v9 = -64;
    d0 = (d0 + -24.0);
    d1 = (d1 + (d5 - d2));
    d2 = (d2 + -1.0);
    d3 = (d3 + (d0 * d3));
    d4 = (d4 + d3);
    printf("%d\n", (((((((((((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) + v9) + arr[(0) & 7]) + arr[(1) & 7]) + arr[(2) & 7]) + arr[(3) & 7]) + arr[(4) & 7]) + arr[(5) & 7]) + arr[(6) & 7]) + arr[(7) & 7]));
}
