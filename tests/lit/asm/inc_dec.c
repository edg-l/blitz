// RUN: %tinyc %s --emit-asm 2>&1
// Simple add/sub should emit add and sub instructions.
// CHECK: add
// CHECK: sub
__attribute__((noinline))
int f(int a, int b) { return (a + b) - 1; }
int main() { return f(10, 11) - 20; }
