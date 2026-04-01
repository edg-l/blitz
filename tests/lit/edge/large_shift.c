// EXIT: 0
// Edge case: shifting by various amounts.
int main() {
    int a = 1;
    if ((a << 0) != 1) { return 1; }
    if ((a << 1) != 2) { return 2; }
    if ((a << 10) != 1024) { return 3; }
    int b = 0 - 1;
    if ((b >> 1) != 0 - 1) { return 5; }
    return 0;
}
