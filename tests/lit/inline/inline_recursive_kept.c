// RUN: %tinyc %s --emit-ir 2>&1
// Recursive function must not be inlined.
// CHECK: function factorial
// CHECK: call factorial
int factorial(int n) {
    if (n < 2) { return 1; }
    return n * factorial(n - 1);
}
int main() { return factorial(5) - 120; }
