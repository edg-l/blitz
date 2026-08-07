// A constant field offset is a displacement, not an index register.
//
// Both stores and the load address the same base with an immediate offset; a
// form that materializes the offset into a register costs a `mov`, a register,
// and an address computation nothing else reads.
struct P { int x; int y; long tag; };

long f(struct P *p) {
    p->tag = 5;
    p->x = 3;
    return p->tag;
}

int main() {
    struct P q;
    return (int)f(&q);
}

// CHECK-LABEL: # main
// CHECK: mov    QWORD PTR [{{[a-z0-9]+}}+0x8],
// CHECK-NOT: [{{[a-z0-9]+}}+{{[a-z0-9]+}}*1]
// EXIT: 5
