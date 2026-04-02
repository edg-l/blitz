// RUN: %tinyc %s --emit-ir 2>&1
// __attribute__((noinline)) must prevent inlining.
// CHECK: function keep_me
// CHECK: call keep_me
__attribute__((noinline))
int keep_me(int x) { return x + 1; }
int main() { return keep_me(41); }
