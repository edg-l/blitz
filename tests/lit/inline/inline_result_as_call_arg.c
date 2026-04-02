// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function result passed directly as argument to a noinline call.
// square(3)=9 should be constfolded; add stays (noinline).
// CHECK: function main
// CHECK: iconst(9
// CHECK: call add

int square(int x) { return x * x; }

__attribute__((noinline))
int add(int a, int b) { return a + b; }

int main() {
    int r = add(square(3), square(4));
    if (r != 25) { return 1; }
    r = add(square(0 - 2), square(0 - 3));
    if (r != 13) { return 2; }
    return 0;
}
