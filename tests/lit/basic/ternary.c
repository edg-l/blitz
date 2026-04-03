// EXIT: 0
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// ternary branches: compare and conditional jump
// CHECK: sub
// CHECK: j

int main() {
    int x = 10;
    // basic ternary
    int a = x > 5 ? 1 : 0;
    if (a != 1) { return 1; }

    // ternary selecting the else branch
    int b = x < 5 ? 100 : 200;
    if (b != 200) { return 2; }

    // nested ternary with distinct comparisons
    // (avoids cross-block flags reuse bug)
    int c = x > 20 ? 1 : x > 8 ? 2 : 3;
    if (c != 2) { return 3; }

    // ternary as abs(x)
    int y = 0 - x;
    int d = y > 0 ? y : 0 - y;
    if (d != 10) { return 4; }

    return 0;
}
