// RUN: %tinyc %s --emit-ir 2>&1
// PASSES: -store-forwarding
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
