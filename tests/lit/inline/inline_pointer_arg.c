// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function that takes and dereferences a pointer.
// read_ptr/write_ptr should be inlined: main has loads/stores directly.
// CHECK: function main
// CHECK-NOT: call read_ptr
// CHECK-NOT: call write_ptr
// CHECK: load I32
// CHECK: store I32

int read_ptr(int *p) { return *p; }

void write_ptr(int *p, int val) { *p = val; }

int main() {
    int x = 10;
    int y = read_ptr(&x);
    if (y != 10) { return 1; }
    write_ptr(&x, 99);
    int z = read_ptr(&x);
    if (z != 99) { return 2; }
    return 0;
}
