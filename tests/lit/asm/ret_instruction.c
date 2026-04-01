// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: ret
int main() { return 42; }
