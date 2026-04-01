// RUN: %tinyc %s --emit-ir 2>&1
// Unsigned rem by power of 2 should become and with mask 7.
// CHECK: iconst(7
// CHECK: x86_and
unsigned urem8(unsigned a) { return a % 8; }
int main() { return (int)urem8(43) - 3; }
