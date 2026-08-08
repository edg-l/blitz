// EXIT: 45
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s

// A loop closes on its own test, not on an unconditional jump.
//
// The header is copied onto the back edge, so what every iteration executes is
// the body and one conditional branch back. The header itself stays where it
// is and runs once, as the guard that a `while` whose trip count can be zero
// needs.
//
// Both nesting levels are checked because the two senses of the header's
// conditional are both reachable here: the trace lays the inner loop's body
// after its header and the outer loop's exit after its own, so one header
// branches out of its loop and the other branches into it. The back edge has to
// carry whichever sense reaches the body, and the CFG is what says which
// successor that is.

// CHECK-LABEL: # main

// The outer guard, run once.
// CHECK: cmp    {{[a-z0-9]+}},0xa
// CHECK: jl

// The inner guard.
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jge

// The inner back edge: the test is at the bottom and the branch back is the
// conditional, with no unconditional jump closing the loop.
// CHECK: cmp    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: jl
// CHECK-NOT: jmp

// The outer back edge, conditional for the same reason.
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
