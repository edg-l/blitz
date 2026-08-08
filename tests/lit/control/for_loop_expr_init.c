// EXIT: 35
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// CHECK: cmp
// The body is laid after the header, so the conditional leaves the loop and
// the fallthrough enters it.
// CHECK: jge
// CHECK: add    {{[a-z0-9]+}},{{[a-z0-9]+}}
// The header's test is copied onto the back edge, so the loop closes on the
// conditional and no unconditional jump closes it.
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jl
// CHECK-NOT: jmp

// init clause is an expression, not a declaration
int main() {
    int i = 0;
    int sum = 0;
    for (i = 5; i < 10; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}
