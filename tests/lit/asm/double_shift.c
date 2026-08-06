// Two shifts of *different* values whose amounts sum to the operand width are a
// double shift: `shld` reads both and writes one, where the pair costs two
// shifts and an or.
// RUN: %tinyc %s --emit-asm 2>&1
// CHECK-LABEL: # funnel32
// CHECK: shld   {{[a-z0-9]+}},{{[a-z0-9]+}},0x7
// CHECK-NOT: shl
// CHECK-LABEL: # funnel64
// CHECK: shld   {{[a-z0-9]+}},{{[a-z0-9]+}},0x14
// CHECK-NOT: shl
// CHECK-LABEL: # funnel_written_right
// CHECK: shld   {{[a-z0-9]+}},{{[a-z0-9]+}},0x9
// CHECK-LABEL: # not_a_funnel
// CHECK: sar
// CHECK-NOT: shld
// EXIT: 55
__attribute__((noinline))
unsigned funnel32(unsigned hi, unsigned lo) { return (hi << 7) | (lo >> 25); }

__attribute__((noinline))
unsigned long funnel64(unsigned long hi, unsigned long lo) {
    return (hi << 20) | (lo >> 44);
}

// The same value with the shifts written the other way round: an Or does not
// record which operand the source put first.
__attribute__((noinline))
unsigned funnel_written_right(unsigned hi, unsigned lo) {
    return (lo >> 23) | (hi << 9);
}

// A signed right shift is not a funnel shift: the sign bit feeds the high end.
__attribute__((noinline))
int not_a_funnel(int hi, int lo) { return (hi << 7) | (lo >> 25); }

int main() {
    unsigned hi = 305419896;
    unsigned lo = 2596069104;
    if (funnel32(hi, lo) != 439041101) { return 1; }
    if (funnel_written_right(hi, lo) != 1756164405) { return 2; }
    if (funnel64(1234567, 9223372036854775807) != 1294537850879) { return 3; }
    if (not_a_funnel(1000, 100000000) != 128002) { return 4; }
    return 55;
}
