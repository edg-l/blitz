// EXIT: 15
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// pointer-based loop using index; the sum is a three-operand lea
// CHECK: lea    {{[a-z0-9]+}},[{{[a-z0-9]+}}+{{[a-z0-9]+}}*1]
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
