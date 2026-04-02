// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: idiv
__attribute__((noinline))
int f(int a, int b) { return a / b; }
int main() { return f(10, 3); }
