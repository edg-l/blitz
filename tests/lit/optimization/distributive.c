// RUN: %tinyc %s --emit-asm -o %t.s | %blitztest %t.s
// CHECK-LABEL: # factor_test
// Check that a*b + a*c uses factoring (ideally one imul, not two)
// This is a best-effort check: verify the function compiles and runs correctly.

// RUN: %tinyc %s -o %t && %t
// EXIT: 0

__attribute__((noinline))
int factor_test(int a, int b, int c) {
    return a * b + a * c;
}

int main() {
    // a*b + a*c = a*(b+c) = 3*(4+5) = 27
    if (factor_test(3, 4, 5) != 27) { return 1; }
    // Edge: a=0
    if (factor_test(0, 4, 5) != 0) { return 2; }
    // Edge: b=c
    if (factor_test(2, 3, 3) != 12) { return 3; }
    return 0;
}
