// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: shl
__attribute__((noinline))
int mul_by_8(int x) { return x * 8; }
int main() { return mul_by_8(5); }
