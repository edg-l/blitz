// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test comma operator precedence interactions.

int main() {
    int a = 10;
    int b = 20;

    // With parens, comma IS the operator: y gets 20 (rightmost).
    int y = (a, b);
    if (y != 20) {
        return 1;
    }

    // Left-associativity: (1, 2, 3) evaluates left to right, returns 3.
    int z = (1, 2, 3);
    if (z != 3) {
        return 2;
    }

    // Nested: (a, (b, a)) returns a (10).
    int w = (a, (b, a));
    if (w != 10) {
        return 3;
    }

    // Comma doesn't leak into assignment RHS:
    // `int v = 42;` then `v = (5, 10);` gives v=10.
    int v = 42;
    v = (5, 10);
    if (v != 10) {
        return 4;
    }

    return 0;
}
