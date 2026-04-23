// RUN: %tinyc %s --disable-inlining --emit-ir 2>&1 | %blitztest %s
// RUN: %tinyc %s -o %t && %t
// EXIT: 120

// A load between two stores observes the first store's value; DSE must NOT
// eliminate the first store.

int save_and_overwrite(int* p, int* q, int y) {
    *p = 100;
    *q = *p + 20;  // observes *p = 100, writes *q = 120
    *p = y;        // overwrites *p, but first store was already observed
    return 0;
}

int main() {
    int x;
    int y;
    save_and_overwrite(&x, &y, 99);
    return y;  // 120
}

// CHECK-LABEL: function save_and_overwrite
// Both stores to *p must survive, plus the store to *q: at least two stores.
// CHECK: store
// CHECK: store
