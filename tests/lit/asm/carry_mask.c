// The 0/-1 mask of an unsigned comparison is the carry flag broadcast over a
// register, which `sbb r, r` does in one instruction. The select form needs
// both constants in registers, a `cmov` and a subtract.
//
// Only the unsigned-below compare is the carry flag: a signed `<` keeps the
// `cmov`, and so does `>`, whose carry would need the compare's operands the
// other way round.
// RUN: %tinyc %s --emit-asm 2>&1
// CHECK-LABEL: # umask
// CHECK: sbb    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK-NOT: cmov
// CHECK-LABEL: # umask64
// CHECK: sbb    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK-NOT: cmov
// CHECK-LABEL: # smask
// CHECK: cmovl
// CHECK-NOT: sbb
// CHECK-LABEL: # umask_above
// CHECK: cmova
// CHECK-NOT: sbb
// EXIT: 7
__attribute__((noinline))
unsigned umask(unsigned a, unsigned b) { return -(unsigned)(a < b); }

__attribute__((noinline))
unsigned long umask64(unsigned long a, unsigned long b) {
    return -(unsigned long)(a < b);
}

// A signed compare is not the carry flag.
__attribute__((noinline))
int smask(int a, int b) { return -(a < b); }

// `a > b` sets the carry flag only for `cmp b, a`, which is a different compare.
__attribute__((noinline))
unsigned umask_above(unsigned a, unsigned b) { return -(unsigned)(a > b); }

int main() {
    int r = 0;
    if (umask(3, 9) != (unsigned)-1) { return 1; }
    if (umask(9, 3) != 0) { return 2; }
    if (umask64(3, 9) != (unsigned long)-1) { return 3; }
    if (umask64(9, 3) != 0) { return 4; }
    if (smask(-5, 1) != -1) { return 5; }
    if (smask(1, -5) != 0) { return 6; }
    if (umask_above(9, 3) != (unsigned)-1) { return 7; }
    if (umask_above(3, 9) != 0) { return 8; }
    r = 7;
    return r;
}
