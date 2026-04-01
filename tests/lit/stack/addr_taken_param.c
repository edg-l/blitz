// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int use_addr(int x) {
    int *p = &x;
    return *p;
}
int main() { return use_addr(42); }
