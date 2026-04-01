// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: stack_addr
// CHECK: store
// CHECK: load
int main() {
    int x = 42;
    int *p = &x;
    return *p;
}
