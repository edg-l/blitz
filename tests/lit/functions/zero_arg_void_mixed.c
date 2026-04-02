// RUN: %tinyc %s --emit-asm 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Regression test: zero-arg void calls interleaved with non-void calls.
// Zero-arg void calls emit no VoidCallBarrier (empty operands), so the
// call point falls back to the arg-scanning heuristic. This must not
// confuse call-point detection for the non-void calls that do have args.
//
// Verify: at least 16 calls (8 nop + 8 id), all id() results compared.
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: sub

__attribute__((noinline))
void nop() { return; }

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
