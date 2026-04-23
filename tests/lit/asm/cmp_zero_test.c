// RUN: %tinyc %s --emit-asm 2>&1
// Comparison with zero should use sub-then-jne (sub sets flags).
// __attribute__((noinline)) prevents inlining + constant-folding.
// CHECK-LABEL: # is_nonzero
// CHECK: cmp
// CHECK: jne
__attribute__((noinline))
int is_nonzero(int x) {
    if (x != 0) { return 1; }
    return 0;
}
int main() { return is_nonzero(5) - 1; }
