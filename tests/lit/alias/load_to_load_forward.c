// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 12

// Load-to-load forwarding: two consecutive loads from the same address with
// no intervening aliasing store — the second load should be eliminated.

int g(int* p) { return *p + *p; }

int main() {
    int x = 6;
    return g(&x);     // *p + *p -> second *p is load-to-load forwarded
}

// CHECK-LABEL: function g
// The function body reads *p twice but only one real load effectful op
// should survive. Advance past the single load, then assert no second load
// before the terminator.
// CHECK: load I32
// CHECK-NOT: load I32
// CHECK: ret
