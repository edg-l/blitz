// RUN: %tinyc %s --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// PASSES: -inlining
// EXIT: 41

// Two fields of one struct are two locations. `p->a` and `p->b` sit at
// constant displacements four bytes apart off the same base, so a store to one
// cannot reach the other and the value stored into `p->a` still forwards to the
// load of it.
//
// Without offsets in the alias model any write to a base invalidated every
// cached load at that base, so a struct with two fields defeated both the
// forwarding pass and dead store elimination -- the reason `alias.rs` splits an
// address into a base expression plus a constant displacement.
//
// Both loads forward, so no load survives into the IR at all.
// CHECK-LABEL: function f
// CHECK-NOT: load

struct P {
    int a;
    int b;
};

int f(struct P* p, int x) {
    p->a = x;
    p->b = x + 1;
    return p->a + p->b;
}

int main() {
    struct P p;
    return f(&p, 20);
}
