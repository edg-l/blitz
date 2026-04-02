// RUN: %tinyc %s --emit-ir 2>&1
// Mul by 5 should become lea with scale 4 via strength reduction + LEA.
// CHECK: x86_lea3(scale=4)
__attribute__((noinline))
int mul5(int a) { return a * 5; }
int main() { return mul5(3) - 15; }
