// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: x86_imul3
int mul(int a, int b) { return a * b; }
int main() { return mul(6, 7); }
