// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 6

// Store-to-load forwarding: store a value to a slot, immediately load it back.
// The load should be eliminated and replaced by the stored value.

int g(int x) { return x + 1; }

int main() {
    int a[2];
    a[0] = g(5);       // a[0] = 6
    a[1] = a[0];       // store-to-load forward: a[1] = a[0] = 6
    return a[1];
}

// CHECK-LABEL: function main
// With forwarding the second load (from a[0]) must be gone: only two stores
// and one call, no load.
// CHECK-NOT: load
// CHECK: ret
