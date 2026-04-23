// EXIT: 10
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop bound fused into cmp via X86CmpI
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jl
// increment inside body
// CHECK: add
// CHECK: jmp

int main() {
    int sum = 0;
    for (int i = 0; i < 10;) {
        sum = sum + 1;
        i = i + 1;
    }
    return sum;
}
