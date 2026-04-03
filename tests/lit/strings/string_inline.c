// EXIT: 0
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// greet() is inlined, string should still work
// CHECK-NOT: call greet
// CHECK: lea
// CHECK: call

extern int puts(char* s);
int greet() {
    puts("inlined hello");
    return 42;
}
int main() {
    int x = greet();
    return x - 42;
}
