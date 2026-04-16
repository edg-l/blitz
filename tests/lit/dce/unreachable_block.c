// RUN: %tinyc %s --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 0

// Test that DCE folds a constant branch and removes the dead else path.
// x=1, so if(x) always takes the true branch.

int main() {
    int x = 1;
    if (x) {
        return 0;
    } else {
        return 1;
    }
}

// The branch should be folded to a jump (no branch instruction).
// CHECK-LABEL: function main
// CHECK-NOT: branch
// The dead else path (return 1) should be eliminated.
// Only one ret should remain.
// CHECK-COUNT-1: ret
