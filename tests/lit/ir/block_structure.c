// RUN: %tinyc %s --emit-ir 2>&1
// CHECK-LABEL: function main
// CHECK: block0
// CHECK: branch
// CHECK: block1
// CHECK: block2
int main() {
    int x = 5;
    if (x > 3) { return 1; }
    return 0;
}
