// EXIT: 0
// Test IEEE 754 float negation: -(+0.0) should be -0.0

int main() {
    double pos_zero = 0.0;
    double neg_result = -pos_zero;

    // -0.0 == +0.0 in IEEE (they compare equal)
    if (neg_result != 0.0) { return 1; }

    // But 1.0 / -0.0 should be -infinity, while 1.0 / +0.0 is +infinity
    double neg_inf = 1.0 / neg_result;
    double pos_inf = 1.0 / pos_zero;
    // -inf < 0 and +inf > 0
    if (!(neg_inf < 0.0)) { return 2; }
    if (!(pos_inf > 0.0)) { return 3; }

    // Basic negation
    double x = 3.14;
    if (-x != 0.0 - 3.14) { return 4; }
    if (-(-x) != 3.14) { return 5; }

    // Float (f32) negation
    float f = 1.5f;
    float nf = -f;
    if (nf != 0.0f - 1.5f) { return 6; }

    return 0;
}
