// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: lea
int f(int a, int b) { return a + b * 2; }
int main() { return f(1, 2); }
