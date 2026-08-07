// A load's address is a GPR however the loaded value is classed.
//
// `LoadResult(_, F64)` reports `is_fp_op()`, which describes its *result*; its
// operands are the folded `Addr` and the base and index the barrier repeats for
// liveness, and those are addresses. The register-class map stamped XMM onto
// all three, so the array's base address was given an XMM register and the
// address left the integer file entirely:
//
//   xor    ebx,ebx           ; i = 0
//   lea    rbx,[rsp+0x8]     ; the array base -- same register, i is gone
//   mov    edx,ebx           ; the counter reads the base
//   cmp    edx,0x8           ; i < 8
//
// `i` then started at the low 32 bits of a stack address. At eight elements
// both loops were skipped and the sum came out zero; at 256 the same code
// indexed far out of bounds, and whether it faulted depended on where ASLR had
// put the stack -- one binary gave a wrong answer on some runs and a segfault
// on others.
//
// The shape it takes: a `double` array and two separate loops over it, so the
// base is live across both while each loop defines its own counter. The same
// program over an `int` array never had it, nor did the fused single-loop
// version, and `-O0` was correct throughout.

// OUTPUT: 28
// EXIT: 0

extern int printf(char* fmt, ...);

int main() {
    double x[8];
    for (int i = 0; i < 8; i = i + 1) {
        x[i] = (double)i;
    }
    double s = 0.0;
    for (int i = 0; i < 8; i = i + 1) {
        s = s + x[i];
    }
    printf("%d\n", (int)s);
    return 0;
}
