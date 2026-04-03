// EXIT: 5
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop comparison
// CHECK: sub
// break jumps to exit
// CHECK: jmp

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
