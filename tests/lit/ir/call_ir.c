// RUN: %tinyc %s --emit-ir 2>&1
// CHECK-LABEL: function main
// CHECK: call foo
// CHECK: call_result
int foo(int x) { return x; }
int main() { return foo(42); }
