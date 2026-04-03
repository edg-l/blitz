// EXIT: 27
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// loop structure
// CHECK: sub
// CHECK: jl
// continue jumps back to header
// CHECK: jmp

// sum 0..9, skipping multiples of 3
int main() {
    int sum = 0;
    for (int i = 0; i < 10; i = i + 1) {
        if (i == 3 || i == 6 || i == 9) {
            continue;
        }
        sum = sum + i;
    }
    // 0+1+2+4+5+7+8 = 27
    return sum;
}
