// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: function main
// CHECK: global_addr(".L.str.0")
// CHECK: global_addr(".L.str.1")
// CHECK-NOT: global_addr(".L.str.2")
// CHECK-NOT: global_addr(".L.str.3")
// CHECK-NOT: global_addr(".L.str.4")
// CHECK: ret

// Test string dedup edge cases

extern int puts(char* s);
int main() {
    // Empty string
    puts("");
    puts("");
    
    // Same non-empty string
    puts("test");
    puts("test");
    
    return 0;
}
