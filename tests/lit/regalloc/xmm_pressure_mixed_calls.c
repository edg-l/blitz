// Regression: an XMM value live across a call must be routed through a slot.
//
// Compilation used to abort in the register allocator:
//
//   phase 'regalloc': register pressure overshoot for function 'main'
//   (gpr_overshoot=0, xmm_overshoot=1). The split pass should have resolved
//   all register pressure before phase 5.
//
// Only 16 XMM registers exist and nothing here is close to that, so it was
// never real pressure. Three defects compounded:
//
// 1. The splitter built its VReg -> RegClass map per block, from that block's
//    instructions alone. Here the double is defined in block1, used in block3,
//    and merely passes through block2 which holds a call -- so it appears in no
//    block2 instruction, had no entry in block2's map, and every pressure count
//    silently skipped it. The allocator uses a function-wide map and saw it
//    interfering with the call-clobber phantoms. The splitter was measuring a
//    different graph than the one being colored.
//
// 2. split.rs modelled call-crossing pressure for GPRs only. Every XMM register
//    is caller-saved, so the callee-saved budget for that class is zero and any
//    XMM value still live after a call needs a slot.
//
// 3. Victim selection called op.result_type(&[]) to test for a Flags-typed def,
//    which asserts arity and panicked on any FP or ALU def (fixed in a90f86d).
//
// Minimal: it needs all three XMM-producing shapes at once. Any two compile
// even with the bugs present:
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

extern int printf(char* fmt, ...);

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
