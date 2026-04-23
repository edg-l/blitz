// EXIT: 55
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop init: i = 1
// CHECK: mov    {{[a-z0-9]+}},0x1
// loop comparison against the 10 bound (immediate fused into cmp via X86CmpI)
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jle
// loop body: add
// CHECK: add
// loop increment: i + 1
// CHECK: add
// backward jump to loop header
// CHECK: jmp

int main() {
    int sum = 0;
    for (int i = 1; i <= 10; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}
