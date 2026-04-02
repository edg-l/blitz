// EXIT: 0
// Test multiple comparison operators on the same operands.
__attribute__((noinline))
int compare(int a, int b) {
    if (a < b) { return 1; }
    if (a == b) { return 2; }
    if (a > b) { return 3; }
    return 4;
}
int main() {
    if (compare(1, 5) != 1) { return 1; }
    if (compare(5, 5) != 2) { return 2; }
    if (compare(9, 5) != 3) { return 3; }
    return 0;
}
