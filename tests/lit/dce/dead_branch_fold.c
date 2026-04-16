// RUN: %tinyc %s --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// OUTPUT: 42

// Test constant branch folding: if(1) folds to unconditional jump,
// eliminating the else block entirely.

extern int printf(char* fmt, int x);

int main() {
    if (1) {
        printf("%d\n", 42);
    } else {
        printf("%d\n", 99);
    }
    return 0;
}

// After branch folding, block0 should jump unconditionally (no branch).
// CHECK-LABEL: function main
// CHECK: jump block
// The dead else block (with 99) should not appear.
// CHECK-NOT: iconst(99
// Only one call to printf should remain.
// CHECK-COUNT-1: call printf
