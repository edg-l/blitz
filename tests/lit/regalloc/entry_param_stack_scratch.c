// A stack-passed parameter's load must not scratch an argument register that
// still holds a parameter.
//
// Nine doubles: eight arrive in XMM0-XMM7 and the ninth on the stack. At -O0
// every value goes to a frame slot, and the load of the ninth needs a register
// to land in -- `pick` handed it XMM0, which still held p0, whose own store came
// later in schedule order and wrote the ninth's value into p0's slot. `p0`
// compared unequal and the function returned 1.
//
// The printf is what makes the failure visible rather than merely present: it
// stands between the entry and the checks, so the sum cannot be folded and the
// parameters have to survive a call.

extern int printf(char* fmt, ...);

__attribute__((noinline)) int nine(double p0, double p1, double p2, double p3,
                                   double p4, double p5, double p6, double p7,
                                   double p8) {
    printf("in\n");
    if (p0 != 1.0) { return 1; }
    if (p1 != 2.0) { return 2; }
    if (p2 != 3.0) { return 3; }
    if (p3 != 4.0) { return 4; }
    if (p4 != 5.0) { return 5; }
    if (p5 != 6.0) { return 6; }
    if (p6 != 7.0) { return 7; }
    if (p7 != 8.0) { return 8; }
    if (p8 != 9.0) { return 9; }
    return 0;
}

int main() {
    printf("%d\n", nine(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    return 0;
}

// EXIT: 0
// OUTPUT: in
// OUTPUT: 0
