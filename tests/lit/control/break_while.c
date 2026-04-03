// EXIT: 7
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// while loop comparison
// CHECK: sub
// break comparison (i == 5)
// CHECK: sub
// CHECK: je
// backward jump for loop
// CHECK: jmp

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
