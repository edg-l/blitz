// EXIT: 45
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s

// A loop closes on its own test, not on an unconditional jump.
//
// The header is copied onto the back edge, so what every iteration executes is
// the body and one conditional branch back. The header itself stays where it
// is and runs once, as the guard that a `while` whose trip count can be zero
// needs.
//
// Both nesting levels are checked, because a nested loop is what makes the
// layout trace's loop depth load-bearing: the depth has to say that the outer
// loop's body outranks its exit, or the trace follows the exit and the outer
// header's conditional goes back on the taken side.

// CHECK-LABEL: # main

// The outer guard, run once, leaving the loop.
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jge

// The inner guard, the same way round.
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jge

// The inner back edge: the test is at the bottom and the branch back is the
// conditional.
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jl

// The outer back edge, conditional for the same reason. Neither loop closes on
// an unconditional jump; the two that remain are the branches *out*, to blocks
// the trace laid before them.
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jl

int main() {
    int total = 0;
    int i = 0;
    while (i < 10) {
        int j = 0;
        while (j < i) {
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
