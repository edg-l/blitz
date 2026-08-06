// A pair of shifts whose amounts sum to the operand width is a rotate, in one
// instruction rather than three. A right rotate arrives as the same Or of two
// shifts, so it is selected as the complementary `rol`.
// RUN: %tinyc %s --emit-asm 2>&1
// CHECK-LABEL: # rotl7
// CHECK: rol    {{[a-z0-9]+}},0x7
// CHECK-NOT: shl
// CHECK-LABEL: # rotr3
// CHECK: rol    {{[a-z0-9]+}},0x1d
// CHECK-NOT: shr
// CHECK-LABEL: # not_a_rotate
// CHECK: sar
// CHECK-NOT: rol
// EXIT: 216
__attribute__((noinline))
unsigned rotl7(unsigned x) { return (x << 7) | (x >> 25); }

__attribute__((noinline))
unsigned rotr3(unsigned x) { return (x >> 3) | (x << 29); }

// A signed right shift is not a rotate: the sign bit feeds the high end.
__attribute__((noinline))
int not_a_rotate(int x) { return (x << 7) | (x >> 25); }

int main() {
    unsigned seed = 305419896;
    if (not_a_rotate(1000) != 128000) { return 1; }
    return (int)((rotl7(seed) + rotr3(seed)) & 255);
}
