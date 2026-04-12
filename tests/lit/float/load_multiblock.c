// Test: float loaded in one block used in multiple successor blocks
// Regression: LoadResult(_, F64) missing from is_fp_op(), causing cross-block
// splitting to use GPR spill/reload instead of XMM, losing the float value
// EXIT: 0

int main() {
    double x = 2.5;
    double *p = &x;
    double val = *p;
    int flag = 1;
    if (flag) {
        if (val > 2.0) {
            return 0;
        }
        return 2;
    }
    return 1;
}
