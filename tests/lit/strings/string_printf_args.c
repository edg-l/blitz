// EXIT: 0
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// string address via RIP-relative LEA
// CHECK: lea
// call to printf
// CHECK: call

extern int printf(char* fmt, ...);
int main() {
    printf("%d + %d = %d\n", 10, 20, 30);
    return 0;
}
