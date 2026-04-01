// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: shl
int foo(int a, int b) { return a + b * 4; }
int main() { return foo(1, 2); }
