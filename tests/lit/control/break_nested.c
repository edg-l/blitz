// EXIT: 6
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// outer loop comparison
// CHECK: sub
// CHECK: jl
// inner loop comparison
// CHECK: sub
// CHECK: jl
// break jump
// CHECK: jmp

// break only exits the inner loop
int main() {
    int count = 0;
    for (int i = 0; i < 3; i = i + 1) {
        for (int j = 0; j < 10; j = j + 1) {
            if (j == 2) {
                break;
            }
            count = count + 1;
        }
    }
    // inner loop runs 2 iterations (j=0,1) x 3 outer = 6
    return count;
}
