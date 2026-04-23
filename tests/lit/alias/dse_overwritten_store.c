// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 42

// Dead store elimination: the first store to *p is dead because the second
// store covers it with no intervening load or call that may read p.

int overwrite(int* p, int y) {
    *p = 99;
    *p = y;
    return 0;
}

int main() {
    int x;
    overwrite(&x, 42);
    return x;
}

// CHECK-LABEL: function overwrite
// Only one store should survive — advance past it then assert no second store.
// CHECK: store
// CHECK-NOT: store
// CHECK: ret
