// RUN: %tinyc %s --emit-asm 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Regression test: 8+ sequential calls where results are compared after.
// The allocator's spill/remat pass must not rematerialize call-arg Iconst
// VRegs away from their call-arg operand position on CallResult, which
// would shorten the live range past call clobber points.
//
// CHECK-LABEL: # main
// CHECK-COUNT-8: call
// Advance past last call, then verify 8 sub+jne comparison pairs.
// CHECK: call
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: sub    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne

__attribute__((noinline))
int id(int x) { return x; }

int main() {
    int a = id(1);
    int b = id(2);
    int c = id(3);
    int d = id(4);
    int e = id(5);
    int f = id(6);
    int g = id(7);
    int h = id(8);
    if (a != 1) { return 1; }
    if (b != 2) { return 2; }
    if (c != 3) { return 3; }
    if (d != 4) { return 4; }
    if (e != 5) { return 5; }
    if (f != 6) { return 6; }
    if (g != 7) { return 7; }
    if (h != 8) { return 8; }
    return 0;
}
