// RUN: %tinyc %s --emit-asm 2>&1
// LEA should also fire for I32 operands (extended in session 004).
// CHECK: lea
__attribute__((noinline))
int f(int a, int b) { return a + b * 4; }
int main() { return f(1, 3) - 13; }
