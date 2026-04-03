// EXIT: 35
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// CHECK: sub
// CHECK: jl
// CHECK: add
// CHECK: jmp

// init clause is an expression, not a declaration
int main() {
    int i = 0;
    int sum = 0;
    for (i = 5; i < 10; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}
