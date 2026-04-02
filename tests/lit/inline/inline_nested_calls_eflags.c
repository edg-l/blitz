// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Stress test: inlined + noinline calls interleaved, comparisons after.
// double_it should be inlined, negate and add3 stay as calls.
// CHECK: function main
// CHECK-NOT: call double_it
// CHECK: call negate
// CHECK: call add3

int double_it(int x) { return x + x; }

__attribute__((noinline))
int negate(int x) { return 0 - x; }

__attribute__((noinline))
int add3(int a, int b, int c) { return a + b + c; }

int main() {
    int a = negate(5);
    int b = double_it(a);
    int c = negate(b);
    int d = double_it(c);
    int e = add3(a, c, d);
    if (a != 0 - 5) { return 1; }
    if (b != 0 - 10) { return 2; }
    if (c != 10) { return 3; }
    if (d != 20) { return 4; }
    if (e != 25) { return 5; }
    return 0;
}
