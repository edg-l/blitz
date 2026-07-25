// KNOWN FAILING -- regression target for a real bug. Do not "fix" by
// weakening the test.
//
// Compilation aborts in the register allocator:
//
//   phase 'regalloc': global regalloc: register pressure overshoot for
//   function 'main' (gpr_overshoot=0, xmm_overshoot=1). The split pass
//   should have resolved all register pressure before phase 5.
//
// The splitter is supposed to have driven pressure below the register budget
// before the allocator runs, so reaching phase 5 over budget means the split
// pass missed a live range. Only 16 XMM registers exist and nothing here is
// remotely close to that, so this is a splitter defect, not real pressure.
//
// ── Diagnosis ────────────────────────────────────────────────────────────────
//
// Two gaps compound. In the IR for main (`--emit-ir`), the double produced in
// block1 is used in block3 and is live *through* block2, which contains a call:
//
//   block1:  v1 = x86_cvtsi2sd(I32)(v0)
//   block2:  call printf(v2, v3)          <- v1 live across this call
//   block3:  v7 = x86_addsd(v1, v6)
//
// 1. src/compile/split.rs models call-crossing pressure for GPRs only:
//    `callee_saved_budget = gpr_budget - CALLER_SAVED_GPR.len()`, then
//    `find_call_crossing_overshoot(.., callee_saved_budget)` hardcoded to
//    RegClass::GPR. Every XMM register is caller-saved, so the XMM budget is
//    zero -- any XMM value live across a call must go through a slot -- but no
//    XMM equivalent of that check exists. Only XMM *block params* are handled,
//    by detect_blockparam_call_crossings (Phase 6).
//
// 2. Adding the missing XMM check is not sufficient on its own (tried: it
//    regressed 6 lit tests). Victim selection works within the block holding
//    the overshoot, and v1 has neither a def nor a use in block2, so there is
//    nothing there to rewrite. The fix has to place the spill store in the
//    defining block and the reload in each using block, which is what
//    SplitScope::CrossBlock is for -- so the gap is in how victims are chosen
//    for a live-through value, not just in the pressure count.
//
// The allocator models the constraint correctly (clobber phantoms pre-colored
// to all 16 XMM registers at each call point, see add_clobber_interferences in
// src/regalloc/allocator.rs), which is why the chromatic number comes out at
// 17 against a budget of 16.
//
// Minimal: it needs all three XMM-producing shapes at once. Any two of them
// compile fine:
//   - a call returning double whose argument is an int  (from_param)
//   - a call returning double taking mixed int/double   (mixed)
//   - an in-line char -> double conversion              (d = c)
//
// Found while writing tests/lit/float/int_to_double_conversions.c.
//
// OUTPUT: a: 10.000000
// OUTPUT: c: 15.700000
// OUTPUT: d: -5.000000
// EXIT: 0

extern int printf(char* fmt, double x);

double from_param(int n) {
    return n;
}

double mixed(int n, double x) {
    return n + x;
}

int main() {
    printf("a: %f\n", from_param(10));
    printf("c: %f\n", mixed(10, 5.7));

    char c;
    c = -5;
    double d;
    d = c;
    printf("d: %f\n", d);
    return 0;
}
