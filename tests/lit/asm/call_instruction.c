// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: call
int foo(int x) { return x; }
int main() { return foo(42); }
