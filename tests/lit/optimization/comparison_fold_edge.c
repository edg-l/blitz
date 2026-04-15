// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test comparison folding edge cases.

int main() {
    int x = 10;

    // Always-true: x <= x
    int a = 0;
    if (x <= x) { a = 1; }
    if (a != 1) { return 1; }

    // Always-false: x > x
    int b = 1;
    if (x > x) { b = 0; }
    if (b != 1) { return 2; }

    // Always-true: x >= x
    int c = 0;
    if (x >= x) { c = 1; }
    if (c != 1) { return 3; }

    // Constant comparison: 10 == 10
    int d = 0;
    if (10 == 10) { d = 1; }
    if (d != 1) { return 4; }

    // Constant comparison: 5 != 10
    int e = 0;
    if (5 != 10) { e = 1; }
    if (e != 1) { return 5; }

    // Constant comparison: unsigned 0 < 1
    int f = 0;
    if (0 < 1) { f = 1; }
    if (f != 1) { return 6; }

    return 0;
}
