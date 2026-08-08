// EXIT: 55
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop init: i = 1
// CHECK: mov    {{[a-z0-9]+}},0x1
// loop comparison against the 10 bound (immediate fused into cmp via X86CmpI).
// The body is laid after the header, so the conditional leaves the loop and
// the fallthrough enters it.
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jg
// loop body: the sum accumulated in place, no copy to set up the addend
// CHECK: add    {{[a-z0-9]+}},{{[a-z0-9]+}}
// loop increment: i + 1, in place
// CHECK: inc    {{[a-z0-9]+}}
// backward jump to loop header
// CHECK: jmp

int main() {
    int sum = 0;
    for (int i = 1; i <= 10; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}
