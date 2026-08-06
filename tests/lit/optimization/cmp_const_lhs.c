// RUN: %tinyc %s --emit-asm -o %t.s | %blitztest %t.s
// RUN: %tinyc %s -o %t && %t
// EXIT: 2

// A constant on the left of a comparison becomes a `cmp r, imm` with the
// condition flipped, not a materialized constant compared against a register.
// Both functions below emit the same two instructions.

// CHECK-LABEL: # const_lhs
// CHECK-NOT: mov    {{[a-z0-9]+}},0x5
// CHECK: cmp    {{[a-z0-9]+}},0x5
// CHECK: jg

// CHECK-LABEL: # const_rhs
// CHECK: cmp    {{[a-z0-9]+}},0x5
// CHECK: jg

__attribute__((noinline))
int const_lhs(int x) {
    if (5 < x) { return 1; }
    return 0;
}

__attribute__((noinline))
int const_rhs(int x) {
    if (x > 5) { return 1; }
    return 0;
}

int main(int argc, char **argv) {
    return const_lhs(argc + 5) + const_rhs(argc + 5);
}
