// EXIT: 10
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop bound: 10
// CHECK: mov    {{[a-z0-9]+}},0xa
// comparison
// CHECK: cmp
// CHECK: jl
// loop body: add 1
// CHECK: add

int main() {
    int i = 0;
    int sum = 0;
    for (; i < 10; i = i + 1) {
        sum = sum + 1;
    }
    return sum;
}
