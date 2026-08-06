// RUN: %tinyc %s --emit-ir 2>&1
// Unsigned rem by power of 2 should become and with mask 7, and the mask goes
// in the instruction rather than a register of its own.
// CHECK: x86_and_imm(7)
__attribute__((noinline))
unsigned urem8(unsigned a) { return a % 8; }
int main() { return (int)urem8(43) - 3; }
