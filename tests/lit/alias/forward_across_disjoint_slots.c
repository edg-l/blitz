// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 9

// A non-aliasing store to a different stack slot must NOT invalidate a
// cached store for the first slot. Using two `int` locals (distinct
// StackAddr(N) classes) lets the alias analyzer prove non-aliasing.

int g(int x) { return x + 1; }

int main() {
    // A single call produces a runtime value (so nothing constant-folds
    // the whole sequence away). Two distinct stack-slot arrays with no
    // intervening call: the store to b[0] must NOT invalidate the cached
    // entry for a[0], so the load of a[0] can still be forwarded.
    int a[1];
    int b[1];
    int v = g(8);   // runtime value, v = 9
    a[0] = v;
    b[0] = v + 100; // store to a different slot
    return a[0];    // forward to v -> 9
}

// CHECK-LABEL: function main
// Both stores survive but the load of a[0] is forwarded away.
// CHECK: store
// CHECK: store
// CHECK-NOT: load
// CHECK: ret
