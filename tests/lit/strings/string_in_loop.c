// EXIT: 0
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// same string literal reused across loop iterations
// CHECK: lea
// CHECK: call

extern int printf(char* fmt, int x);
int main() {
    for (int i = 0; i < 3; i = i + 1) {
        printf("i=%d\n", i);
    }
    return 0;
}
