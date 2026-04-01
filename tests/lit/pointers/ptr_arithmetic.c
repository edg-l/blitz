// RUN: %tinyc %s -o %t && %t
// EXIT: 42
// Test pointer arithmetic: write via p[0] and read back
int main() {
    int x = 0;
    int *p = &x;
    *p = 42;
    int *q = p + 0;
    return *q;
}
