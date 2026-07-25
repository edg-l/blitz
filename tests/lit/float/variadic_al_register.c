// Regression: SysV requires AL to hold the number of vector registers used to
// pass arguments. A variadic callee branches on AL to decide whether to spill
// XMM0-7 into its register save area, so if AL is zero the callee reads a save
// area that was never written.
//
// Blitz never set AL. Every passing `printf("%f", ...)` test did so by luck,
// because materializing an int argument had left a nonzero value in EAX. This
// case has no such accident: the only argument is a long converted to double,
// so RAX held 0x12a05f200 whose low byte is 0x00, and printf printed garbage.
//
// Blitz cannot tell a variadic callee from a fixed one (tinyc prototypes carry
// no `...`), so it sets AL on every call. A fixed callee ignores it.
//
// OUTPUT: big: 5000000000.000000
// EXIT: 0

extern int printf(char* fmt, double x);

double from_long(long n) {
    return n;
}

int main() {
    printf("big: %f\n", from_long(5000000000));
    return 0;
}
