// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: x86_movsx
// Use a parameter so the value isn't constant-folded away.
__attribute__((noinline))
int sext_test(char c) {
    return (int)c;
}
int main() {
    return sext_test(42);
}
