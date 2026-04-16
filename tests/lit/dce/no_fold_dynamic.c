// RUN: %tinyc %s --emit-ir 2>&1 | %blitztest %s

// Test that DCE does NOT fold a branch with a truly dynamic condition
// (function call result is not known at compile time).

int getval();

int main() {
    int x = getval();
    if (x) {
        return 1;
    } else {
        return 0;
    }
}

// The call result is unknown; branch must remain.
// CHECK-LABEL: function main
// CHECK: branch
