// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
//
// A multiply by a constant no shift-and-lea decomposition reaches is the
// 3-operand `imul`, which is the one immediate form that is not two-address:
// it reads its operand and writes a different register, so neither the
// `mov r, imm32` for the constant nor the `mov dst, src` for the destructive
// form is emitted.
//
// CHECK-LABEL: # main
// CHECK-NOT: mov    {{e[a-z0-9]+}},0x186a0
// CHECK-DAG: imul   {{[a-z0-9]+}},{{[a-z0-9]+}},0x186a0
// CHECK-DAG: imul   {{[a-z0-9]+}},{{[a-z0-9]+}},0x35
//
// Constants a shift or an LEA can build still go that way; the multiply is
// what is left over.
// CHECK-NOT: imul   {{[a-z0-9]+}},{{[a-z0-9]+}},0x3
int printf(char *fmt, ...);

int main(int argc) {
    int wide = argc * 100000;
    int narrow = argc * 53;
    int shifted = argc * 8;
    int lea3 = argc * 3;
    printf("%d %d %d %d\n", wide, narrow, shifted, lea3);
    return 0;
}
