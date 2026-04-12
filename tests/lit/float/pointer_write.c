// Test: write through float pointer (store path)
// Known bug: OpSize::from_type panics on F64 in effectful.rs Store path
// EXIT: 42

int main() {
    double x = 0.0;
    double *p = &x;
    *p = 42.5;
    int result = (int)x;
    return result;
}
