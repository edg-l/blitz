// EXIT: 0
// Test ++/-- operators (prefix and postfix)

int main() {
    // Pre-increment
    int a = 5;
    int b = ++a;
    if (a != 6) { return 1; }
    if (b != 6) { return 2; }

    // Post-increment
    int c = 10;
    int d = c++;
    if (c != 11) { return 3; }
    if (d != 10) { return 4; }

    // Pre-decrement
    int e = 5;
    int f = --e;
    if (e != 4) { return 5; }
    if (f != 4) { return 6; }

    // Post-decrement
    int g = 10;
    int h = g--;
    if (g != 9) { return 7; }
    if (h != 10) { return 8; }

    // In for-loop update
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += i;
    }
    if (sum != 10) { return 9; }

    // Decrement in condition
    int count = 0;
    int j = 5;
    while (j > 0) {
        j--;
        count++;
    }
    if (count != 5) { return 10; }
    if (j != 0) { return 11; }

    // Chained with expressions
    int x = 3;
    int y = ++x * 2;
    if (x != 4) { return 12; }
    if (y != 8) { return 13; }

    return 0;
}
