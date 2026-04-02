// EXIT: 0
// Test comparisons between negative numbers.
__attribute__((noinline))
int neg_compare(int a, int b) {
    if (a < b) { return 1; }
    if (a == b) { return 2; }
    return 3;
}
int main() {
    if (neg_compare(0 - 10, 0 - 5) != 1) { return 1; }
    if (neg_compare(0 - 5, 0 - 5) != 2) { return 2; }
    if (neg_compare(0 - 1, 0 - 5) != 3) { return 3; }
    return 0;
}
