// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function with many arguments (tests param remapping).
// First two calls const-fold after inlining (15 and 150).
// CHECK: function main
// CHECK: iconst(-15
// CHECK: iconst(15
// CHECK: iconst(150

int sum5(int a, int b, int c, int d, int e) {
    return a + b + c + d + e;
}

int main() {
    int r = sum5(1, 2, 3, 4, 5);
    if (r != 15) { return 1; }
    int s = sum5(10, 20, 30, 40, 50);
    if (s != 150) { return 2; }
    int t = sum5(0 - 1, 0 - 2, 0 - 3, 0 - 4, 0 - 5);
    if (t != 0 - 15) { return 3; }
    return 0;
}
