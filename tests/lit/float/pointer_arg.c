// Test: float function args passed through pointers (load from pointer, pass to noinline fn)
// Regression: OpSize::from_type panicked on F64 in Load effectful lowering
// EXIT: 5

__attribute__((noinline)) int double_to_int(double v) {
    return (int)v;
}

int main() {
    double x = 5.9;
    double *p = &x;
    double loaded = *p;
    int result = double_to_int(loaded);
    return result;
}
