// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// EXIT: 0
// CHECK: lea
// CHECK: call
// CHECK: lea
// CHECK: call
extern int puts(char* s);
int main() {
    puts("first string");
    puts("second string");
    return 0;
}
