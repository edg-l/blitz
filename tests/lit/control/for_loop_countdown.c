// EXIT: 10
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// counting down from 10 to 0, decrement as `dec` -- the constant is in the
// instruction rather than a register of its own
// CHECK: mov    {{[a-z0-9]+}},0xa
// CHECK: dec
// backward jump
// CHECK: jmp

int main() {
    int sum = 0;
    for (int i = 10; i > 0; i = i - 1) {
        sum = sum + 1;
    }
    return sum;
}
