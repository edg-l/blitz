// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function with many intermediate computations.
// compute(1,2,3)=47 should be constfolded after inlining.
// CHECK: function main
// CHECK-NOT: function compute
// CHECK: iconst(47

int compute(int a, int b, int c) {
    int x = a + b;
    int y = b + c;
    int z = a + c;
    int p = x * y;
    int q = y * z;
    int r = x * z;
    return p + q + r;
}

int main() {
    int r1 = compute(1, 2, 3);
    if (r1 != 47) { return 1; }
    int r2 = compute(0, 0, 0);
    if (r2 != 0) { return 2; }
    int r3 = compute(10, 20, 30);
    if (r3 != 4700) { return 3; }
    return 0;
}
