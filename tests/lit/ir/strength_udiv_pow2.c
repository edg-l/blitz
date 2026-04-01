// RUN: %tinyc %s --emit-ir 2>&1
// Unsigned div by power of 2 should become shr.
// CHECK: x86_shr_imm(3)
unsigned udiv8(unsigned a) { return a / 8; }
int main() { return (int)udiv8(40) - 5; }
