// Test: float pointer swap (read from pointer then write back different value)
// Regression: OpSize::from_type panicked on F64 in Load/Store effectful lowering
// EXIT: 10

int main() {
    double a = 3.0;
    double b = 10.0;
    double *pa = &a;
    double *pb = &b;
    double tmp = *pa;
    *pa = *pb;
    *pb = tmp;
    // a == 10.0, b == 3.0
    int result = (int)*pa;
    return result;
}
