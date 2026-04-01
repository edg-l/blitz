// RUN: %tinyc %s --emit-ir 2>&1
// Idempotence: a & a = a, a | a = a should not emit and/or ops.
// CHECK-NOT: x86_and
// CHECK-NOT: x86_or
int idem_and(int a) { return a & a; }
int main() { return idem_and(7) - 7; }
