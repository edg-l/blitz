// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: stack_addr
// CHECK: store
// CHECK: call puts
extern int puts(char *s);
int main() {
    puts("test");
    return 0;
}
