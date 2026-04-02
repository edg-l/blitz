// RUN: %tinyc %s --emit-asm 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test void calls with multiple arguments: VoidCallBarrier must keep
// all arg VRegs live at the call point.
//
// Verify: 3 void set() calls emitted, followed by sub comparisons.
// CHECK: call
// CHECK: call
// CHECK: call
// CHECK: sub
// CHECK: sub
// CHECK: sub

__attribute__((noinline))
void set(int *p, int v) { *p = v; }

int main() {
    int a = 0;
    int b = 0;
    int c = 0;
    set(&a, 10);
    set(&b, 20);
    set(&c, 30);
    if (a != 10) { return 1; }
    if (b != 20) { return 2; }
    if (c != 30) { return 3; }
    return 0;
}
