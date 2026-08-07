// A mask that only clears bits the shift already zeroed is not an instruction:
// known-bits propagation gives `Shr(x, n)` zeros in its top n bits and `Shl(x, n)`
// zeros in its low n bits, and the redundant-And rule then merges the mask away.
// A mask that clears a bit the shift left unknown must survive.
// RUN: %tinyc %s --emit-asm 2>&1
// CHECK-LABEL: # hi_byte
// CHECK: shr    {{[a-z0-9]+}},0x18
// CHECK-NOT: and
// CHECK-LABEL: # hi_word
// CHECK: shr    {{[a-z0-9]+}},0x10
// CHECK-NOT: and
// CHECK-LABEL: # low_zeroed
// CHECK: shl    {{[a-z0-9]+}},0x8
// CHECK-NOT: and
// CHECK-LABEL: # mid_byte
// CHECK: shr    {{[a-z0-9]+}},0x8
// CHECK: and

__attribute__((noinline))
int hi_byte(unsigned int x) { return (int)((x >> 24) & 255); }

__attribute__((noinline))
int hi_word(unsigned int x) { return (int)((x >> 16) & 65535); }

// The shift zeroed bits 0..7, so clearing them again is redundant.
__attribute__((noinline))
int low_zeroed(unsigned int x) { return (int)((x << 8) & 4294967040); }

// Bits 8..31 of the shifted value are unknown, so this mask is real work.
__attribute__((noinline))
int mid_byte(unsigned int x) { return (int)((x >> 8) & 255); }

int main(int argc, char **argv) {
    return hi_byte(argc) + hi_word(argc) + low_zeroed(argc) + mid_byte(argc);
}
