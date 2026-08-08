// EXIT: 15
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// The array base is loop-invariant and hoisted; the scaled index rides in the
// addressing mode of the access itself.
// CHECK: lea    {{[a-z0-9]+}},[rsp{{[-+]0x[0-9a-f]+}}]
// CHECK: mov    {{[a-z0-9]+}},DWORD PTR [{{[a-z0-9]+}}+{{[a-z0-9]+}}*4]
// CHECK: jmp

int main() {
    int arr[5];
    int* p = arr;
    for (int i = 0; i < 5; i = i + 1) {
        *(p + i) = i + 1;
    }
    int sum = 0;
    for (int i = 0; i < 5; i = i + 1) {
        sum = sum + *(p + i);
    }
    return sum;
}
