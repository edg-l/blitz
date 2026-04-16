// RUN: %tinyc %s --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 42

// Test constant branch folding for comparison operators.
// 5 > 3 is always true, so the else branch should be eliminated.

int main() {
    if (5 > 3) {
        return 42;
    } else {
        return 0;
    }
}

// The comparison folds to true; no branch in the IR.
// CHECK-LABEL: function main
// CHECK-NOT: branch
// CHECK: iconst(42
// CHECK: ret
