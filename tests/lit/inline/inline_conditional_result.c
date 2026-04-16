// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function with conditional return, used in further arithmetic.
// abs_val(-42) and abs_val(42) both constfold to 42 after inlining.
// CHECK: function main
// CHECK: iconst(42
// CHECK: iconst(50

int abs_val(int x) {
    if (x < 0) { return 0 - x; }
    return x;
}

int max(int a, int b) {
    if (a > b) { return a; }
    return b;
}

int min(int a, int b) {
    if (a < b) { return a; }
    return b;
}

int main() {
    int a = abs_val(0 - 42);
    int b = abs_val(42);
    if (a != b) { return 1; }
    int c = max(a, 100);
    if (c != 100) { return 2; }
    int d = min(a, 100);
    if (d != 42) { return 3; }
    int e = max(min(200, 50), abs_val(0 - 30));
    if (e != 50) { return 4; }
    return 0;
}
