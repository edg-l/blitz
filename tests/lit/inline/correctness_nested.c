// RUN: %tinyc %s -o %t && %t
// EXIT: 120

// End-to-end correctness: factorial via chained helpers, all inlined.

int mul(int a, int b) {
    return a * b;
}

int fac_step(int acc, int n) {
    return mul(acc, n);
}

int main() {
    int result = 1;
    int i = 1;
    while (i <= 5) {
        result = fac_step(result, i);
        i = i + 1;
    }
    return result;
}
