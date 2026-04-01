// RUN: %tinyc %s --emit-ir 2>&1
// Identity rules: a + 0 = a, a * 1 = a should be folded away.
// The IR should NOT contain add or imul for these trivial cases.
// CHECK-NOT: x86_add
// CHECK-NOT: x86_imul
int identity(int a) {
    int b = a + 0;
    int c = b * 1;
    return c;
}
int main() { return identity(42) - 42; }
