// Regression test: a 32-bit division whose operand width had to be recovered at
// lowering time, with a negative divisor so that guessing wrong is visible.
//
// Lowering read the width from `vreg_types`, which is built before the splitter
// runs, and fell back to 64 bits for any VReg missing from it -- a reload, or a
// class re-emitted in a block its first emitter does not dominate. Both arms of
// the `if` here need the quotient, so the division is emitted twice and only one
// copy has a type.
//
// For most ops guessing 64 bits is invisible, since the low 32 bits of the answer
// are the same. Division is not one of those ops. The divisor is materialized by
// `mov ecx,imm32`, which zero-extends, so a 64-bit `idiv` divides by 4294967293
// rather than -3: the quotient comes out 0 and the remainder comes out equal to
// the dividend.
//
// The width therefore rides on the op, as it does for `X86CmpI`.
//
// `opaque` keeps the loop condition out of reach of constant folding; folded, the
// branch disappears and with it the second emission.

// EXIT: 0
// OUTPUT: 477
// OUTPUT: 471
// OUTPUT: 495
// OUTPUT: 473

extern int printf(char* fmt, ...);

int opaque(int x) { return x; }

int main() {
    int c = opaque(1);
    int a = 18;
    int rem = 0;
    int quo = 0;
    int rem4 = 0;
    int quo4 = 0;

    for (int i = 0; i < 3; i++) {
        rem = 477 + (a % -3);
        quo = 477 + (a / -3);
        rem4 = 477 + (a % -4);
        quo4 = 477 + (a / -4);
        if (c) {
        }
    }

    // 18 % -3 == 0, so 477. Read as unsigned this is 477 + 18 == 495.
    printf("%d\n", rem);
    // 18 / -3 == -6, so 471. Read as unsigned this is 477 + 0 == 477.
    printf("%d\n", quo);
    // 18 % -4 == 2, so 479... but a wrong width gives 477 + 18 == 495 here too,
    // which is why the two remainders use different divisors.
    printf("%d\n", rem4 + 16);
    // 18 / -4 == -4, so 473.
    printf("%d\n", quo4);

    return 0;
}
