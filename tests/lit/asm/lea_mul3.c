// RUN: %tinyc %s --emit-asm 2>&1
// Mul by 3 via strength reduction should emit lea.
// CHECK: lea
__attribute__((noinline))
int mul3(int a) { return a * 3; }
int main() { return mul3(4) - 12; }
