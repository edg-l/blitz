// Test: float pointer dereference (load through float pointer)
// EXIT: 0

int main() {
    double x = 3.14;
    double *p = &x;
    double y = *p;
    if (y > 3.0) {
        return 0;
    }
    return 1;
}
