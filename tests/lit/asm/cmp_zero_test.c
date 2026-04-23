// RUN: %tinyc %s --emit-asm 2>&1
// Comparison with zero lowers through X86CmpI(0) which emits a flag-only
// `test r, r` — same flags as `cmp r, 0`, 1 byte shorter.
// __attribute__((noinline)) prevents inlining + constant-folding.
// CHECK-LABEL: # is_nonzero
// CHECK: test
// CHECK: jne
__attribute__((noinline))
int is_nonzero(int x) {
    if (x != 0) { return 1; }
    return 0;
}
int main() { return is_nonzero(5) - 1; }
