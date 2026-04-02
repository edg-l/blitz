// EXIT: 0
// Test x < 0 with multiple comparisons on same operands.
__attribute__((noinline))
int classify(int x) {
    if (x < 0) { return 1; }
    if (x == 0) { return 2; }
    return 3;
}
int main() {
    if (classify(0 - 5) != 1) { return 1; }
    if (classify(0) != 2) { return 2; }
    if (classify(5) != 3) { return 3; }
    return 0;
}
