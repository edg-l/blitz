// Regression: constant-index array stores in same block as a call.
// The barrier scheduler must schedule StackAddr and Addr nodes before
// the stores that use them, even when the same StackAddr is also a
// call argument at a later barrier position.
// RUN: %tinyc %s --emit-asm | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 42

// CHECK-LABEL: # main
// The StackAddr LEA must appear before any stores to the array.
// CHECK: lea    {{[a-z0-9]+}},[rsp]
// CHECK: mov    DWORD PTR
// CHECK: mov    DWORD PTR
// CHECK: mov    DWORD PTR
// CHECK: mov    DWORD PTR
// CHECK: call

__attribute__((noinline))
int identity(int *p) {
    return p[0];
}

int main() {
    int arr[4];
    arr[0] = 10;
    arr[1] = 12;
    arr[2] = 20;
    arr[3] = 42;
    return identity(arr + 3);
}
