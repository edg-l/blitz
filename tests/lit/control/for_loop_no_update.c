// EXIT: 10
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop bound fused into cmp via X86CmpI
// CHECK: cmp    {{[a-z0-9]+}},0xa
// The body is laid after the header, so the conditional leaves the loop and
// the fallthrough enters it.
// CHECK: jge
// increment inside body, the constant a lea displacement
// CHECK: inc    {{[a-z0-9]+}}
// The header's test is copied onto the back edge, so the loop closes on the
// conditional and no unconditional jump closes it.
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jl
// CHECK-NOT: jmp

int main() {
    int sum = 0;
    for (int i = 0; i < 10;) {
        sum = sum + 1;
        i = i + 1;
    }
    return sum;
}
