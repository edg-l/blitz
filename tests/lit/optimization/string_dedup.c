// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: function main
// CHECK: global_addr(".L.str.0")
// CHECK-NOT: global_addr(".L.str.1")
// CHECK-NOT: global_addr(".L.str.2")
// CHECK-NOT: global_addr(".L.str.3")
// CHECK: ret

// Verify string deduplication: "hello" appears 3 times but only one .L.str.0 is emitted

extern int puts(char* s);
int main() {
    puts("hello");
    puts("hello");
    puts("hello");
    return 0;
}
