// EXIT: 2
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// Use noinline to keep the comparisons dynamic (parameter-based).
// CHECK-LABEL: # check
// block0: first x > 5 comparison
// CHECK: sub
// CHECK: jg
// block1: x > 20 comparison (clobbers EFLAGS)
// CHECK: sub
// CHECK: jg
// block3: fresh x > 5 comparison (must not reuse stale flags from block0)
// CHECK: sub
// CHECK: jg

__attribute__((noinline))
int check(int x) {
    if (x > 5) {
        if (x > 20) {
            return 1;
        }
        if (x > 5) {
            return 2;
        }
    }
    return 3;
}
int main() { return check(10); }
