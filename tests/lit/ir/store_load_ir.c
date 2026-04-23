// RUN: %tinyc %s --disable-store-forwarding --emit-ir 2>&1
// Verifies the raw shape of stack_addr / store / load IR construction.
// Store-to-load forwarding is disabled so the load is not eliminated.
// CHECK: stack_addr
// CHECK: store
// CHECK: load
int main() {
    int x = 42;
    int *p = &x;
    return *p;
}
