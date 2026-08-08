// EXIT: 15
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop comparison against bound 5 (immediate fused into cmp via X86CmpI)
// CHECK: cmp    {{[a-z0-9]+}},0x5
// The body is laid after the header, so the conditional leaves the loop and
// the fallthrough enters it.
// CHECK: jge
// The scaled index is the store's own addressing mode, not a separate `lea`
// into a register the store then ignores.
// CHECK: mov    DWORD PTR [{{[a-z0-9]+}}+{{[a-z0-9]+}}*4]
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
