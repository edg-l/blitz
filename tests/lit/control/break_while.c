// EXIT: 7
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// while loop comparison
// CHECK: cmp
// break comparison (i == 5). The block the branch falls into is the rest of the
// body, so the conditional is the break itself and it leaves the loop.
// CHECK: cmp    {{[a-z0-9]+}},0x5
// CHECK: je
// The back edge is the loop's own test, so the loop closes on no
// unconditional jump.
// CHECK: cmp
// CHECK: jl
// CHECK-NOT: jmp

// break and continue in while loops
int main() {
    int i = 0;
    int sum = 0;
    while (i < 100) {
        if (i == 5) {
            break;
        }
        sum = sum + i;
        i = i + 1;
    }
    // 0+1+2+3+4 = 10, i == 5 at exit
    // 10 + 5 - 8 = 7
    return sum + i - 8;
}
