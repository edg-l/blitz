// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function called repeatedly inside a loop.
// triple() should be fully inlined and dead-eliminated.
// CHECK: function main
// CHECK-NOT: function triple
// CHECK-NOT: call triple

int triple(int x) { return x + x + x; }

int main() {
    int acc = 0;
    int i = 1;
    while (i <= 5) {
        acc = acc + triple(i);
        i = i + 1;
    }
    if (acc != 45) { return 1; }
    return 0;
}
