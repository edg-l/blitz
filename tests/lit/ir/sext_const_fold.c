// RUN: %tinyc %s --emit-ir 2>&1
// CHECK: iconst(42
// CHECK-NOT: x86_movsx
// Known-bits + constant promotion folds (int)(char)42 to iconst(42) at IR level.
int main() {
    char c = 42;
    return (int)c;
}
