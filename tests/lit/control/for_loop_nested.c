// EXIT: 45
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// outer loop bound
// CHECK: mov    {{[a-z0-9]+}},0x3
// CHECK: sub
// CHECK: jl
// inner loop bound
// CHECK: mov    {{[a-z0-9]+}},0x3
// CHECK: sub
// CHECK: jl
// i*3 via lea
// CHECK: lea

int main() {
    int sum = 0;
    for (int i = 0; i < 3; i = i + 1) {
        for (int j = 0; j < 3; j = j + 1) {
            sum = sum + (i * 3 + j + 1);
        }
    }
    return sum;
}
