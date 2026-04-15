// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test cross-rule optimization interactions end-to-end.

// Factoring + strength: a*4 + a*8 = a*12
__attribute__((noinline))
int factor_then_strength(int a) {
    return a * 4 + a * 8;
}

// Comparison fold + select: always-true branch selected, always-false eliminated
__attribute__((noinline))
int comparison_cascade(int x) {
    int a = (x == x) ? 10 : 20;   // always 10
    int b = (3 < 5) ? a : 99;     // always a = 10
    return b;
}

// Known-bits through shifts: constant discovery
__attribute__((noinline))
int shift_constant_discovery(int x) {
    int a = 1 << 3;   // 8
    int b = 1 << 2;   // 4
    return x + a + b;  // x + 12
}

// Factoring in subtraction context
__attribute__((noinline))
int factor_sub_then_strength(int a) {
    return a * 8 - a * 4;  // a * 4
}

int main() {
    // factor_then_strength(3) = 3*4 + 3*8 = 12 + 24 = 36
    if (factor_then_strength(3) != 36) { return 1; }
    if (factor_then_strength(0) != 0) { return 2; }
    if (factor_then_strength(-1) != -12) { return 3; }

    // comparison_cascade always returns 10
    if (comparison_cascade(42) != 10) { return 4; }
    if (comparison_cascade(0) != 10) { return 5; }

    // shift_constant_discovery(100) = 100 + 12 = 112
    if (shift_constant_discovery(100) != 112) { return 6; }
    if (shift_constant_discovery(0) != 12) { return 7; }

    // factor_sub_then_strength(5) = 5*8 - 5*4 = 40 - 20 = 20
    if (factor_sub_then_strength(5) != 20) { return 8; }
    if (factor_sub_then_strength(0) != 0) { return 9; }

    return 0;
}
