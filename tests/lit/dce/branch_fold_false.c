// RUN: %tinyc %s --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 20

// Test constant branch folding for the false path: if(0) folds to the else.

int main() {
    if (0) {
        return 99;
    } else {
        return 20;
    }
}

// The branch should be folded; the true path (return 99) eliminated.
// CHECK-LABEL: function main
// CHECK-NOT: branch
// CHECK-NOT: iconst(99
// CHECK: ret
