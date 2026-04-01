// RUN: %tinyc %s --emit-ir 2>&1
// Inverse rules: a - a = 0, a ^ a = 0.
// These should fold to zero.
// CHECK: iconst(0
int sub_self(int a) { return a - a; }
int main() { return sub_self(99); }
