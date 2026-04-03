// EXIT: 6
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// index assign in update clause
// CHECK: lea
// CHECK: jmp

// use arr[0] as the loop counter via index-assign update
int main() {
    int arr[2];
    arr[0] = 0;
    arr[1] = 0;
    for (; arr[0] < 4; arr[0] = arr[0] + 1) {
        arr[1] = arr[1] + arr[0];
    }
    return arr[1];
}
