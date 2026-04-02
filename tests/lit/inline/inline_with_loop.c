// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function containing a while loop.
// sum_to should be inlined and eliminated.
// CHECK: function main
// CHECK-NOT: function sum_to
// CHECK-NOT: call sum_to

int sum_to(int n) {
    int acc = 0;
    while (n > 0) {
        acc = acc + n;
        n = n - 1;
    }
    return acc;
}

int main() {
    int a = sum_to(10);
    if (a != 55) { return 1; }
    int c = sum_to(0);
    if (c != 0) { return 2; }
    return 0;
}
