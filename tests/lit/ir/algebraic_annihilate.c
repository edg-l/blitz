// RUN: %tinyc %s --emit-ir 2>&1
// Annihilation rules: a * 0 = 0, a & 0 = 0.
// These should constant-fold to zero.
// CHECK: iconst(0
// CHECK-NOT: x86_imul
int annihilate(int a) {
    return a * 0;
}
int main() { return annihilate(99); }
