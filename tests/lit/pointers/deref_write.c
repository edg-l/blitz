// RUN: %tinyc %s -o %t && %t
// EXIT: 99
int main() {
    int x = 0;
    int *p = &x;
    *p = 99;
    return x;
}
