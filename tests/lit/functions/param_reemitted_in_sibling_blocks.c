// Regression test: a parameter read after a branch must keep one VReg.
//
// `p0` is not used in the entry block, so VReg linearization used to emit its
// Param op lazily in the first block that reads it. Both arms of the `if` reach
// the return, neither dominates the other, so each got its own
// `Param(0, F64)` VReg -- and only one of the two carried the ABI precolor to
// XMM0. The other was free to land in XMM1, and the phi copy on that edge then
// read a parameter out of a register that never held it:
//
//   test edi,edi
//   jg   .then
//   jmp  .ret
//  .then:
//   movsd xmm0,xmm1      <- "copy" p0 from XMM1, which holds p2
//  .ret:
//   ret
//
// So f1 returned p2 (-364.0) instead of p0 (-14.0) at -O0. -O1 was unaffected,
// which is why the O0-vs-O1 differential harness is what caught it (via
// tests/fuzz/gen_c.py seeds 4 and 6).
//
// OUTPUT: -14

extern int printf(char* fmt, ...);

double f1(double p0, int p1, double p2) {
    if (p1 > 0) {
        p1 = 1;
    } else {
        p1 = 2;
    }
    return p0;
}

int main() {
    printf("%d\n", (int)f1(-14.0, 5, -364.0));
    return 0;
}
