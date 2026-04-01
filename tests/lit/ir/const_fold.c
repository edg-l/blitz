// RUN: %tinyc %s --emit-ir 2>&1
// Constant folding: 3 + 4 should fold to iconst(7).
// CHECK: iconst(7
// CHECK-NOT: x86_add
int fold() { return 3 + 4; }
int main() { return fold() - 7; }
