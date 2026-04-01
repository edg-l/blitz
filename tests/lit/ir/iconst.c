// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: iconst(42, I32)
// CHECK: ret
int main() { return 42; }
