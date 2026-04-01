// RUN: %tinyc %s -o %t && %t
// EXIT: 1
int main() {
    int a = 42;
    int b = 42;
    if (a == b) { return 1; }
    return 0;
}
