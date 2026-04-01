// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: x86_sub
// CHECK: branch
int main() {
    int x = 5;
    if (x > 3) { return 1; }
    return 0;
}
