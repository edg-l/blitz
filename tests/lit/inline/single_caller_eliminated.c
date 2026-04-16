// RUN: %tinyc %s --emit-ir 2>&1
// At O1 (default), a single-caller leaf is inlined; helper disappears.
// CHECK-LABEL: function main
// CHECK-NOT: function helper

// RUN: %tinyc %s -o %t && %t
// EXIT: 7

int helper(int x) {
    return x + 2;
}

int main() {
    return helper(5);
}
