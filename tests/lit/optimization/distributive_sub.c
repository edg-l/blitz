// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test distributive factoring for subtraction: a*b - a*c = a*(b-c)

__attribute__((noinline))
int factor_sub(int a, int b, int c) {
    return a * b - a * c;
}

__attribute__((noinline))
int factor_add_commuted(int a, int b, int c) {
    // b*a + c*a = a*(b+c) — factor on right side
    return b * a + c * a;
}

int main() {
    // a*(b-c) = 3*(7-4) = 9
    if (factor_sub(3, 7, 4) != 9) { return 1; }
    // Edge: result is negative
    if (factor_sub(3, 4, 7) != -9) { return 2; }
    // Edge: b == c, result is 0
    if (factor_sub(5, 3, 3) != 0) { return 3; }
    // Edge: a == 0
    if (factor_sub(0, 7, 3) != 0) { return 4; }

    // Commuted: b*a + c*a = a*(b+c) = 2*(3+4) = 14
    if (factor_add_commuted(2, 3, 4) != 14) { return 5; }
    // Edge: a == 1
    if (factor_add_commuted(1, 10, 20) != 30) { return 6; }

    return 0;
}
