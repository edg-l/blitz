// EXIT: 0
// Test complex float expressions, mixed int/float, and edge cases.

int main() {
    // Float arithmetic chain
    double a = 1.5;
    double b = 2.5;
    double c = a * b + a - b;  // 3.75 + 1.5 - 2.5 = 2.75
    int ci = (int)(c * 100.0);  // 275
    if (ci != 275) { return 1; }

    // Mixed int/float promotion
    int x = 3;
    double y = x + 0.5;
    int yi = (int)(y * 10.0);
    if (yi != 35) { return 2; }

    // Negation chain
    double v = 42.0;
    if (-(-v) != 42.0) { return 3; }
    if (-(-(-v)) != -42.0) { return 4; }

    // Float in ternary
    double t = (a > b) ? a : b;
    int ti = (int)(t * 10.0);  // 25
    if (ti != 25) { return 5; }

    // f32 arithmetic
    float fa = 1.5f;
    float fb = 2.0f;
    float fc = fa * fb;  // 3.0
    if (fc != 3.0f) { return 6; }

    return 0;
}
