// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// The meet over all predecessors: both arms of the diamond pass 7, so the
// parameter is that constant, goes away, and `x * 3 + 1` folds to 22 rather
// than being computed at run time.
// CHECK-LABEL: function all_arms_agree
// CHECK-NOT: block_param
// CHECK: iconst(22
//
// Different constants meet to nothing, so the parameter stays and so does the
// arithmetic that reads it.
// CHECK-LABEL: function arms_disagree
// CHECK: block_param
// CHECK-NOT: iconst(22
// CHECK-NOT: iconst(28
__attribute__((noinline))
int all_arms_agree(int argc) {
    int x;
    if (argc > 1) { x = 7; } else { x = 7; }
    return x * 3 + 1;
}

__attribute__((noinline))
int arms_disagree(int argc) {
    int x;
    if (argc > 1) { x = 7; } else { x = 9; }
    return x * 3 + 1;
}

int main() {
    if (all_arms_agree(1) != 22) { return 1; }
    if (arms_disagree(2) != 22) { return 2; }
    return 0;
}
