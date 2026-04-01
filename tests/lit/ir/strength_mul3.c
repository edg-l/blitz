// RUN: %tinyc %s --emit-ir 2>&1
// Mul by 3 should become lea with scale 2 via strength reduction + LEA.
// CHECK: x86_lea3(scale=2)
int mul3(int a) { return a * 3; }
int main() { return mul3(7) - 21; }
