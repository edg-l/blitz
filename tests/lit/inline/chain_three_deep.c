// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Three levels of inlining: main -> double_it -> add_one -> (leaf)
// add_one inlined into double_it; first two double_it calls inlined+constfolded.
// CHECK: function main
// CHECK-NOT: function add_one
// CHECK: iconst(12
// CHECK: iconst(2

int add_one(int x) { return x + 1; }
int double_it(int x) { return add_one(x) + add_one(x); }

int main() {
    int r = double_it(5);
    if (r != 12) { return 1; }
    r = double_it(0);
    if (r != 2) { return 2; }
    r = double_it(0 - 3);
    if (r != 0 - 4) { return 3; }
    return 0;
}
