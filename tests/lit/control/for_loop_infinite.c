// EXIT: 5
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// Infinite loop with early return: must have backward jmp and a ret.
// CHECK: je
// CHECK: jmp
// CHECK: ret

int main() {
    int i = 0;
    for (;;) {
        if (i == 5) {
            return i;
        }
        i = i + 1;
    }
    return 0;
}
