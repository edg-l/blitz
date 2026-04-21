// EXIT: 45
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// LICM + iconst dedup hoist the shared bound 0x3 once; both loop compares
// use the same register.
// CHECK: mov    {{[a-z0-9]+}},0x3
// outer loop compare + branch
// CHECK: sub
// CHECK: jl
// inner loop compare + branch
// CHECK: sub
// CHECK: jl
// i*3 via lea (scale-by-3 addressing)
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
