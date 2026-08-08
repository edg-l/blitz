// EXIT: 45
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// Each loop compare against the bound 3 fuses the immediate into `cmp`
// directly via X86CmpI — no separate `mov ..., 0x3` for the bound.
// Each loop's guard, run once. The body is laid after its header at both
// levels -- which is what the layout trace's loop depth has to be exact for --
// so both conditionals leave their loop and both fallthroughs enter it.
// CHECK: cmp    {{[a-z0-9]+}},0x3
// CHECK: jge
// inner loop compare + branch
// CHECK: cmp    {{[a-z0-9]+}},0x3
// CHECK: jge
// i*3 via lea (scale-by-3 addressing)
// CHECK: lea

int main() {
    int sum = 0;
    for (int i = 0; i < 3; i = i + 1) {
        for (int j = 0; j < 3; j = j + 1) {
            sum = sum + (i * 3 + j + 1);
        }
    }
    return sum;
}
