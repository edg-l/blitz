// RUN: %tinyc %s -o %t && %t
// EXIT: 77
int main() {
    int a = 0;
    int *p = &a;
    p[0] = 77;
    return a;
}
