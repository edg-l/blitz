// An odd number of stack arguments must not leave RSP misaligned at the call.
//
// SysV AMD64 (§3.2.2) requires RSP % 16 == 0 at a `call`, and the frame layout
// establishes that for the body, so the argument pushes are the only thing that
// can break it: one `push` per stack argument, and an odd count moves RSP by
// eight with nothing putting it back. The callee then hands glibc a misaligned
// stack and the `movaps` in `printf`'s SSE path faults.
//
// Even counts survive by luck, which is why odd ones are the case worth pinning:
// seven and nine arguments here, either side of the eight-argument shape that
// works whether or not the padding exists.

extern int printf(char* fmt, int x);

__attribute__((noinline))
int seven(int a, int b, int c, int d, int e, int f, int g) {
    printf("%d\n", a + b + c + d + e + f + g);
    return 0;
}

__attribute__((noinline))
int eight(int a, int b, int c, int d, int e, int f, int g, int h) {
    printf("%d\n", a + b + c + d + e + f + g + h);
    return 0;
}

__attribute__((noinline))
int nine(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
    printf("%d\n", a + b + c + d + e + f + g + h + i);
    return 0;
}

int main() {
    seven(1, 2, 3, 4, 5, 6, 7);
    eight(1, 2, 3, 4, 5, 6, 7, 8);
    nine(1, 2, 3, 4, 5, 6, 7, 8, 9);
    return 0;
}

// OUTPUT: 28
// OUTPUT: 36
// OUTPUT: 45
// EXIT: 0
