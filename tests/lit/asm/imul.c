// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: imul
int mul(int a, int b) { return a * b; }
int main() { return mul(6, 7); }
