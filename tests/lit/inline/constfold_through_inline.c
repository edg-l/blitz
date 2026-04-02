// RUN: %tinyc %s --emit-ir 2>&1
// Inlining add(3,4) should allow constant folding to 7.
// CHECK: iconst(7
// CHECK-NOT: call add
int add(int a, int b) { return a + b; }
int main() { return add(3, 4); }
