// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 7

// A call between a store and a load acts as a barrier: forwarding must not
// eliminate the load because the call could have written to the slot.

extern int puts(char* s);

int g(int x) { return x + 1; }

int main() {
    int a[1];
    a[0] = g(5);     // a[0] = 6
    puts("x");       // call barrier: any store after here may have modified a[0]
    a[0] = a[0] + 1; // load after the call must NOT be forwarded
    return a[0];     // 7
}

// CHECK-LABEL: function main
// The read of a[0] after puts() must remain as a real load.
// CHECK: load
