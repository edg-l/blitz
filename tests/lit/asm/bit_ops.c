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
// EXIT: 63
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

int main() {
  return setbit(1, 3) + clrbit(15, 1) + flipbit(4, 2) + (int)setbit64(1, 5) +
         setbit_const(0);
}
