// RUN: %tinyc %s --emit-ir 2>&1
// Nested inlining: double(triple(2)) -> triple folds to 6, then double(6) = 12.
// triple(2) = 2+2+2 = 6, double_it(6) = 6+6 = 12, fully folded.
// CHECK: iconst(12
// CHECK-NOT: call
int triple(int x) { return x + x + x; }
int double_it(int x) { return x + x; }
int main() { return double_it(triple(2)); }
