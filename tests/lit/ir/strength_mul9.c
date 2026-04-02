// RUN: %tinyc %s --emit-ir 2>&1
// Mul by 9 should become lea with scale 8 via strength reduction + LEA.
// CHECK: x86_lea3(scale=8)
__attribute__((noinline))
int mul9(int a) { return a * 9; }
int main() { return mul9(2) - 18; }
