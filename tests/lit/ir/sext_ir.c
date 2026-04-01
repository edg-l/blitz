// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: x86_movsx
int main() {
    char c = 42;
    return (int)c;
}
