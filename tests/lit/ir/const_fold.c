// RUN: %tinyc %s --emit-ir 2>&1
// Constant folding: 3 + 4 folds to 7, and the caller's `- 7` then folds with
// it, so `main` is the difference and nothing else.
// CHECK: iconst(0
// CHECK-NOT: x86_add
// CHECK-NOT: x86_sub
int fold() { return 3 + 4; }
int main() { return fold() - 7; }
