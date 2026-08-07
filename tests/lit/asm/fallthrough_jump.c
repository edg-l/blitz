// EXIT: 12
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
//
// The else arm of a ternary jumps to the merge block, and after branch
// threading that block is the one that already follows: a short jump with
// displacement 0, `eb 00`, over no bytes at all. No constant in this function
// encodes those bytes any other way.
//
// CHECK-LABEL: # main
// CHECK-NOT: eb 00

int main(int argc, char **argv) {
    int x = argc + 9;
    int c = x > 20 ? 1 : x > 5 ? 2 : 3;
    int y = 0 - x;
    int d = y > 0 ? y : 0 - y;
    int e = c + d;
    if (e > 100) { return 1; }
    return e;
}
