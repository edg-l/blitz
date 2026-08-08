// EXIT: 5
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// The loop guard, run once, branching out of the loop.
// CHECK: cmp
// CHECK: jge
// The break's own test, which leaves the loop on the same exit.
// CHECK: cmp    {{[a-z0-9]+}},0x5
// CHECK: je
// The back edge is the loop's test, so nothing here jumps unconditionally.
// CHECK: cmp
// CHECK: jl
// CHECK-NOT: jmp

int main() {
    int sum = 0;
    for (int i = 0; i < 100; i = i + 1) {
        if (i == 5) {
            break;
        }
        sum = sum + 1;
    }
    return sum;
}
