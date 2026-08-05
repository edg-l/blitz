// FAILING: an -O1 wrong-value bug. cc prints 28, blitz prints 0.
//
// The loop counter and the array's base address get the same register while
// both are live:
//
//   xor    ebx,ebx           ; i = 0
//   lea    rbx,[rsp+0x8]     ; the array base -- same register, i is gone
//   mov    edx,ebx           ; the counter reads the base
//   cmp    edx,0x8           ; i < 8
//
// So `i` starts at the low 32 bits of a stack address. Here that is always
// above 8, both loops are skipped and the sum is zero. With a larger array the
// same code indexes far out of bounds, and whether it faults depends on where
// ASLR put the stack -- the 256-element form of this program segfaults on some
// runs of one binary and prints a wrong answer on others.
//
// What it needs: a `double` array and two separate loops over it, so the base
// is live across both while each loop defines its own counter. The same program
// over an `int` array is correct, and so is the fused single-loop version.
// Correct at -O0.

// OUTPUT: 28
// EXIT: 0

extern int printf(char* fmt, int x);

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
