// RUN: %tinyc %s --emit-ir 2>&1
// Mul by power of 2 should become shl via strength reduction.
// CHECK: x86_shl_imm(3)
__attribute__((noinline))
int mul8(int a) { return a * 8; }
int main() { return mul8(5) - 40; }
