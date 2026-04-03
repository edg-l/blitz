// RUN: %tinyc %s -o %t && %t
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// EXIT: 18
//
// Regression: inlining a function that accesses a global variable crashed
// with "phi-elim: param class not in class_to_vreg" because block params
// created by inlining were not included as extraction roots.

// CHECK-LABEL: # main
// increment() should be inlined -- no call instructions
// CHECK-NOT: call
// RIP-relative LEA for the global address
// CHECK: lea    {{[a-z0-9]+}},[rip+{{.*}}]
// Two adds for the two increment(5) and increment(3) calls
// CHECK: add    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: add    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: ret

int counter;
int increment(int n) {
    counter = counter + n;
    return counter;
}
int main() {
    counter = 10;
    int a = increment(5);
    int b = increment(3);
    return b;
}
