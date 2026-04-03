// EXIT: 10
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// field assign in update clause
// CHECK: add
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
