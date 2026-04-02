// RUN: %tinyc %s --emit-ir 2>&1
// Mix of inlinable and noinline functions. Only the noinline one survives.
// CHECK-NOT: function small
// CHECK: function big_noinline
// CHECK: call big_noinline
int small(int x) { return x + 1; }
__attribute__((noinline))
int big_noinline(int x) { return x * 2; }
int main() { return big_noinline(small(20)); }
