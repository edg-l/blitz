// Test: float load result used after an integer branch (cross-block float liveness)
// Regression: LoadResult(_, F64) missing from is_fp_op(), cross-block split
// used GPR instead of XMM for float load results
// EXIT: 0

int main() {
    double x = 2.5;
    double *p = &x;
    double val = *p;
    int ok = 1;
    if (ok) {
        if (val > 2.0) {
            if (val < 3.0) {
                return 0;
            }
            return 3;
        }
        return 2;
    }
    return 1;
}
