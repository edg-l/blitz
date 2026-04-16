// RUN: %tinyc %s --enable-licm -o %t && %t
// EXIT: 30

// Nested loop: the inner loop runs 3 times per outer iteration.
// LICM should handle nested loops without crashing or miscompiling.

int main() {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        int j = 0;
        while (j < 3) {
            sum = sum + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}
