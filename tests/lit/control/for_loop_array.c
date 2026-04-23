// EXIT: 15
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop comparison against bound 5 (immediate fused into cmp via X86CmpI)
// CHECK: cmp    {{[a-z0-9]+}},0x5
// CHECK: jl
// array base address via lea
// CHECK: lea
// scaled index access
// CHECK: lea    {{[a-z0-9]+}},{{.*}}
// backward jump for loop iteration
// CHECK: jmp

int main() {
    int arr[5];
    for (int i = 0; i < 5; i = i + 1) {
        arr[i] = i + 1;
    }
    int sum = 0;
    for (int i = 0; i < 5; i = i + 1) {
        sum = sum + arr[i];
    }
    return sum;
}
