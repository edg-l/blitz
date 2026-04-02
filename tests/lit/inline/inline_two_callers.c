// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Two different callers inline the same function.
int clamp_positive(int x) {
    if (x < 0) { return 0; }
    return x;
}
int test_a() { return clamp_positive(10) - 10; }
int test_b() { return clamp_positive(0 - 3); }
int main() { return test_a() + test_b(); }
