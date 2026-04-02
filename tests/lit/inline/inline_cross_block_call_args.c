// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Regression test: inlined constants from different blocks passed as
// call args to noinline functions. The inlined values (inc->11, dbl->40,
// triple->9) are constfolded into separate early blocks, then flow via
// block params into a block with multiple calls. The per-block allocator
// must not assign them colliding registers.
//
// Verify inlining happened: inc/dbl/triple should be gone, constants visible.
// CHECK: function main
// CHECK-NOT: function inc
// CHECK-NOT: function dbl
// CHECK-NOT: function triple
// CHECK: iconst(11
// CHECK: iconst(40

__attribute__((noinline))
int id(int x) { return x; }

int inc(int x) { return x + 1; }
int dbl(int x) { return x * 2; }
int triple(int x) { return x + x + x; }

int main() {
    int a = inc(10);
    int b = dbl(20);
    int c = triple(3);
    int ra = id(a);
    int rb = id(b);
    int rc = id(c);
    if (ra != 11) { return 1; }
    if (rb != 40) { return 2; }
    if (rc != 9) { return 3; }
    return 0;
}
