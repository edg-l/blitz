// EXIT: 0
// Test do-while loops (basic cases without break/continue)

int main() {
    // Basic do-while: executes at least once
    int x = 0;
    do {
        x = x + 1;
    } while (x < 5);
    if (x != 5) { return 1; }

    // do-while executes body even when condition is initially false
    int y = 10;
    do {
        y = y + 1;
    } while (y < 5);
    if (y != 11) { return 2; }

    // Single iteration
    int z = 100;
    do {
        z = z * 2;
    } while (z < 50);
    // z = 200 (body runs once, then 200 < 50 is false)
    if (z != 200) { return 3; }

    return 0;
}
