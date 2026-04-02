// RUN: %tinyc %s --emit-ir 2>&1
// After inlining f(5), the block param should be propagated and 5*3=15 folded.
// CHECK: iconst(15
int f(int x) { return x * 3; }
int main() { return f(5); }
