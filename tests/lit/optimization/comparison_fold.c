// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test that trivially-true/false comparisons are folded.

int main() {
    int x = 42;

    // Same-operand comparison: x == x is always true
    int a = 0;
    if (x == x) {
        a = 1;
    }
    if (a != 1) { return 1; }

    // Constant comparison: 3 < 5 is always true
    int b = 0;
    if (3 < 5) {
        b = 1;
    }
    if (b != 1) { return 2; }

    // Constant comparison: 5 < 3 is always false
    int c = 1;
    if (5 < 3) {
        c = 0;
    }
    if (c != 1) { return 3; }

    return 0;
}
