// A one-bit mask built from a variable index is the single-bit instruction that
// reads the index directly: `bts`/`btr`/`btc` against a constant into a
// register, a shift routed through CL, and the ALU op.
// RUN: %tinyc %s --emit-asm 2>&1
// CHECK-LABEL: # setbit
// CHECK: bts    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK-NOT: shl
// CHECK-LABEL: # clrbit
// CHECK: btr    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK-NOT: not
// CHECK-LABEL: # flipbit
// CHECK: btc    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK-NOT: shl
// CHECK-LABEL: # setbit64
// CHECK: bts    r{{[a-z0-9]+}},r{{[a-z0-9]+}}
// CHECK-LABEL: # setbit_const
// CHECK-NOT: bts
// CHECK: or
// CHECK-LABEL: # setbit_wide
// CHECK: bts    {{[a-z0-9]+}},0x10
// CHECK-LABEL: # clrbit_wide
// CHECK: btr    {{[a-z0-9]+}},0x14
// CHECK-LABEL: # flipbit_wide
// CHECK: btc    {{[a-z0-9]+}},0x10
// CHECK-LABEL: # setbit_wide64
// CHECK-NOT: movabs
// CHECK: bts    r{{[a-z0-9]+}},0x28
// EXIT: 76
__attribute__((noinline))
int setbit(int x, int n) { return x | (1 << n); }

__attribute__((noinline))
int clrbit(int x, int n) { return x & ~(1 << n); }

__attribute__((noinline))
int flipbit(int x, int n) { return x ^ (1 << n); }

__attribute__((noinline))
long setbit64(long x, long n) { long one = 1; return x | (one << n); }

// A constant index folds the mask, and the immediate-form `or` is three bytes
// where `bts` is four.
__attribute__((noinline))
int setbit_const(int x) { return x | (1 << 3); }

// A one-bit mask from bit 7 up is no longer an `imm8`, so the register form has
// to materialize it and the immediate-index bit instruction is shorter.
__attribute__((noinline))
int setbit_wide(int x) { return x | 65536; }

__attribute__((noinline))
int clrbit_wide(int x) { return x & -1048577; }

__attribute__((noinline))
int flipbit_wide(int x) { return x ^ 65536; }

__attribute__((noinline))
long setbit_wide64(long x) { long one = 1; return x | (one << 40); }

int main() {
  return setbit(1, 3) + clrbit(15, 1) + flipbit(4, 2) + (int)setbit64(1, 5) +
         setbit_const(0) + (setbit_wide(1) >> 14) + (clrbit_wide(3) & 7) +
         (flipbit_wide(2) >> 15) + (int)(setbit_wide64(5) >> 38);
}
