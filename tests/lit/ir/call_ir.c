// RUN: %tinyc %s --emit-ir 2>&1
// CHECK-LABEL: function main
// CHECK: call foo
// CHECK: call_result
__attribute__((noinline))
int foo(int x) { return x; }
int main() { return foo(42); }
