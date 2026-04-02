// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: x86_add
__attribute__((noinline))
int add(int a, int b) { return a + b; }
int main() { return add(3, 4); }
