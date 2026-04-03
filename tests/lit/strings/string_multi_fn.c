// EXIT: 0
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// noinline helpers emit first, then main
// CHECK-LABEL: # fn1
// CHECK: lea
// CHECK: call
// CHECK-LABEL: # fn2
// CHECK: lea
// CHECK: call
// CHECK-LABEL: # main
// CHECK: call
// CHECK: call

extern int puts(char* s);
__attribute__((noinline)) void fn1() { puts("fn1 string"); }
__attribute__((noinline)) void fn2() { puts("fn2 string"); }
int main() {
    fn1();
    fn2();
    return 0;
}
