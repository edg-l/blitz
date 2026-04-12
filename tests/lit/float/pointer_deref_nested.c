// Test: float pointer deref with nested if (cross-block float liveness)
// Known bug: loaded float value doesn't survive across block boundaries
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
