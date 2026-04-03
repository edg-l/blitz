// EXIT: 5
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// no comparison -- infinite loop desugars to while(1)
// early return inside loop body
// CHECK: ret
// backward jump for infinite loop
// CHECK: jmp

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
