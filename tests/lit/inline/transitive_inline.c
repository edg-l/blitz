// RUN: %tinyc %s --emit-ir 2>&1
// Bottom-up inlining: C is inlined into B, then B (with C's body) is inlined into main.
// Neither helper function should remain.
// CHECK-LABEL: function main
// CHECK-NOT: call add_one
// CHECK-NOT: call double_add
// CHECK: ret

// RUN: %tinyc %s -o %t && %t
// EXIT: 14

int add_one(int x) {
    return x + 1;
}

int double_add(int x) {
    int a = add_one(x);
    return a + a;
}

int main() {
    return double_add(6);
}
