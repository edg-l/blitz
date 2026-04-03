// EXIT: 0
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// RIP-relative LEA for string literal
// CHECK: lea
// CHECK: call

// string literal stored in a char* variable
extern int puts(char* s);
int main() {
    char* msg = "stored in variable";
    puts(msg);
    return 0;
}
