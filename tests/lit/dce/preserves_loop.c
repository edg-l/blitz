// RUN: %tinyc %s --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 55

// Test that DCE does not eliminate reachable loop blocks.
// Returns the sum 1+2+...+10 = 55 as the exit code.

int main() {
    int sum = 0;
    int i = 1;
    while (i <= 10) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}

// The loop header with the branch must survive DCE.
// CHECK-LABEL: function main
// CHECK: branch
