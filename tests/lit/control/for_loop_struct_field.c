// EXIT: 10
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// field assign in update clause, the increment a lea displacement
// CHECK: lea    {{[a-z0-9]+}},[{{[a-z0-9]+}}+0x1]
// CHECK: jmp

struct Counter {
    int val;
};

int main() {
    struct Counter c;
    c.val = 0;
    int sum = 0;
    for (; c.val < 10; c.val = c.val + 1) {
        sum = sum + 1;
    }
    return sum;
}
