// EXIT: 0
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// constant-folded sum 0xFF + 0x10 + 0x15 = 0x124
// CHECK: mov    {{[a-z0-9]+}},0x124
// 0x1a check
// CHECK: mov    {{[a-z0-9]+}},0x1a

int main() {
    int a = 0xFF;
    int b = 0x10;
    int c = 0xA + 0xB;
    // 255 + 16 + 21 = 292 (0x124)
    int sum = a + b + c;
    if (sum != 292) {
        return 1;
    }
    // test uppercase 0X prefix
    if (0X1A != 26) {
        return 2;
    }
    return 0;
}
