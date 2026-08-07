// RUN: %tinyc %s --emit-asm 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Regression test: zero-arg void calls interleaved with non-void calls.
// Zero-arg void calls emit no VoidCallBarrier (empty operands), so the
// call point falls back to the arg-scanning heuristic. This must not
// confuse call-point detection for the non-void calls that do have args.
//
// CHECK-LABEL: # main
// CHECK-COUNT-17: call
// Advance past last call, then verify sub+jne comparison pairs.
// CHECK: call
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jne

// The store is what keeps this call here at all: a void function that stores
// nothing and calls nothing is pure, its result list is empty, and
// `dce::eliminate_dead_pure_calls` removes every call to it -- correctly, and
// leaving this test asserting nothing about call-point detection. A store makes
// it observable, and the call site is still the zero-argument void call whose
// `VoidCallBarrier` carries no operands, which is the shape under test.
__attribute__((noinline))
void nop() {
    int sink[1];
    sink[0] = 1;
}

__attribute__((noinline))
int id(int x) { return x; }

int main() {
    nop();
    int a = id(1);
    nop();
    int b = id(2);
    nop();
    int c = id(3);
    nop();
    int d = id(4);
    nop();
    int e = id(5);
    nop();
    int f = id(6);
    nop();
    int g = id(7);
    nop();
    int h = id(8);
    nop();
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
