// RUN: %tinyc %s --emit-ir 2>&1
// DCE folds the constant branch (5 > 3 is always true), leaving only the true path.
// CHECK-LABEL: function main
// CHECK: block0
// CHECK: jump
// CHECK: block1
// CHECK: iconst(1
// CHECK: ret
int main() {
    int x = 5;
    if (x > 3) { return 1; }
    return 0;
}
