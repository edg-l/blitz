// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 77

// Three consecutive stores at the same address: the first two are dead.

int three_stores(int* p, int final) {
    *p = 10;
    *p = 20;
    *p = final;
    return 0;
}

int main() {
    int x;
    three_stores(&x, 77);
    return x;
}

// CHECK-LABEL: function three_stores
// Only one store must survive — advance past it and assert no more before ret.
// CHECK: store
// CHECK-NOT: store
// CHECK: ret
