// EXIT: 0
// Test NaN behavior in branch conditions.
// Each test is isolated in a separate noinline function to avoid e-graph class merging.

__attribute__((noinline)) double make_nan() {
    return 0.0 / 0.0;
}

__attribute__((noinline)) int test_nan_eq() {
    double nan = make_nan();
    double x = 5.0;
    if (nan == x) { return 1; }
    return 0;
}

__attribute__((noinline)) int test_nan_ne() {
    double nan = make_nan();
    double x = 5.0;
    if (nan != x) { return 1; }
    return 0;
}

__attribute__((noinline)) int test_nan_lt() {
    double nan = make_nan();
    double x = 5.0;
    if (nan < x) { return 1; }
    return 0;
}

__attribute__((noinline)) int test_nan_gt() {
    double nan = make_nan();
    double x = 5.0;
    if (nan > x) { return 1; }
    return 0;
}

__attribute__((noinline)) int test_nan_le() {
    double nan = make_nan();
    double x = 5.0;
    if (nan <= x) { return 1; }
    return 0;
}

__attribute__((noinline)) int test_nan_ge() {
    double nan = make_nan();
    double x = 5.0;
    if (nan >= x) { return 1; }
    return 0;
}

int main() {
    if (test_nan_eq() != 0) { return 1; }
    if (test_nan_ne() != 1) { return 2; }
    if (test_nan_lt() != 0) { return 3; }
    if (test_nan_gt() != 0) { return 4; }
    if (test_nan_le() != 0) { return 5; }
    if (test_nan_ge() != 0) { return 6; }
    return 0;
}
