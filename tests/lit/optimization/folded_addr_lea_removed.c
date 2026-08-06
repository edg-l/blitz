// RUN: %tinyc %s --emit-asm -o %t.s | %blitztest %t.s
// RUN: %tinyc %s -o %t && %t
// EXIT: 5

// A constant field offset becomes an addressing-mode displacement, and the LEA
// that computed the address is gone: the store folded it into its own operand.
// CHECK-LABEL: # store_tag
// CHECK-NOT: lea
// CHECK: mov    QWORD PTR [{{[a-z0-9]+}}+0x8]

struct P { int x; int y; long tag; };

__attribute__((noinline))
long store_tag(struct P *p) {
    p->tag = 5;
    p->x = 3;
    return p->tag;
}

int main() {
    struct P q;
    return (int)store_tag(&q);
}
