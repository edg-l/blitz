// Test: float pointer dereference (load/store through float pointers)
// Known bug: OpSize::from_type panics on F64/F32 in effectful.rs
// EXIT: 0

int main() {
    double x = 3.14;
    double *p = &x;
    double y = *p;
    if (y > 3.0) {
        if (y < 4.0) {
            return 0;
        }
    }
    return 1;
}
