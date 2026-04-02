// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Regression test: EFLAGS clobbered across 3+ calls in the same block.
// The comparison (sub) for the branch must not be separated from the
// branch by intervening call instructions that clobber EFLAGS.

__attribute__((noinline))
int classify(int x) {
    if (x < 0) { return 0 - 1; }
    if (x == 0) { return 0; }
    return 1;
}

int main() {
    int a = classify(0 - 5);
    int b = classify(0);
    int c = classify(5);
    if (a != 0 - 1) { return 1; }
    if (b != 0) { return 2; }
    if (c != 1) { return 3; }
    return 0;
}
