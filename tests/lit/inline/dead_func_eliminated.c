// RUN: %tinyc %s --emit-ir 2>&1
// After inlining id into main, the id function should be eliminated.
// Only main should remain in the IR output.
// CHECK: function main
// CHECK-NOT: function id
int id(int x) { return x; }
int main() { return id(99); }
