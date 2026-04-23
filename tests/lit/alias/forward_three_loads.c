// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 30

// Three consecutive loads from the same address collapse to one: the
// second and third are load-to-load forwarded.

int triple(int* p) {
    return *p + *p + *p;
}

int main() {
    int x = 10;
    return triple(&x);
}

// CHECK-LABEL: function triple
// Only one load effectful op survives in this function.
// CHECK: load I32
// CHECK-NOT: load I32
// CHECK: ret
