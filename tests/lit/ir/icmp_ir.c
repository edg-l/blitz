// RUN: %tinyc %s --emit-ir 2>&1
// DCE folds the constant branch (5 > 3), so no x86_sub or branch remains.
// CHECK-LABEL: function main
// CHECK-NOT: branch
// CHECK: ret
int main() {
    int x = 5;
    if (x > 3) { return 1; }
    return 0;
}
