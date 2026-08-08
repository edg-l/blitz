// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
//
// An immediate that does not fit in `imm8` still belongs in the instruction.
// The register form would need `mov r, imm32` first, which is what
// `CostModel::operand_needs_register` charges it for, so the `imm32` form wins
// on the seven bytes it saves rather than losing on the three it costs.
//
// CHECK-LABEL: # main
// CHECK-NOT: mov    {{e[a-z0-9]+}},0x1e240
// CHECK-DAG: and    {{[a-z0-9]+}},0x1e240
// CHECK-DAG: xor    {{[a-z0-9]+}},0xf423f
int printf(char *fmt, ...);

int main(int argc) {
    int b = argc & 123456;
    int c = argc ^ 999999;
    printf("%d %d\n", b, c);
    return 0;
}
