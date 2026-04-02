// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined functions with pointer side effects -- ordering must be preserved.
// At least some load_and_bump calls should be inlined (load+store in main).
// CHECK: function main
// CHECK: load I32
// CHECK: store I32

int load_and_bump(int *p) {
    int old = *p;
    *p = old + 1;
    return old;
}

int main() {
    int counter = 0;
    int a = load_and_bump(&counter);
    int b = load_and_bump(&counter);
    int c = load_and_bump(&counter);
    if (a != 0) { return 1; }
    if (b != 1) { return 2; }
    if (c != 2) { return 3; }
    if (counter != 3) { return 4; }
    return 0;
}
