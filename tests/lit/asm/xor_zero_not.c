// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: xor
// CHECK-NOT: mov{{.*}}0x0
int main() { return 0; }
