// Test: read and write through a float pointer, then read back through same pointer
// Regression: OpSize::from_type panicked on F64 in Load/Store effectful lowering
// Writes 1.0, 2.0, 3.0 to same location sequentially and reads back 3.0
// EXIT: 6

int main() {
    double x = 0.0;
    double *p = &x;
    *p = 1.0;
    double a = *p;
    *p = 2.0;
    double b = *p;
    *p = 3.0;
    double c = *p;
    int result = (int)(a + b + c);
    return result;
}
