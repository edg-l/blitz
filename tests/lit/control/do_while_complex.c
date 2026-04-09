// EXIT: 0
// Test do-while in more complex patterns (single variable per loop)

int main() {
    // do-while computing powers of 2 (single variable)
    int val = 1;
    do {
        val = val * 2;
    } while (val < 1024);
    if (val != 1024) { return 1; }

    // do-while single iteration with large value
    int big = 999;
    do {
        big = big + 1;
    } while (big < 100);
    if (big != 1000) { return 2; }

    // do-while countdown
    int n = 10;
    do {
        n = n - 1;
    } while (n > 0);
    if (n != 0) { return 3; }

    return 0;
}
