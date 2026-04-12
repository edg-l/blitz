// Test: float value from pointer load survives across multiple nested block boundaries
// Regression: LoadResult(_, F64) missing from is_fp_op(), cross-block split
// used GPR instead of XMM for float load results, losing the loaded value
// EXIT: 0

int main() {
    double val = 4.5;
    double *p = &val;
    double x = *p;
    if (x > 1.0) {
        if (x > 2.0) {
            if (x > 3.0) {
                if (x > 4.0) {
                    return 0;
                }
                return 4;
            }
            return 3;
        }
        return 2;
    }
    return 1;
}
