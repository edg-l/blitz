// RUN: %tinyc %s -o %t && %t
// EXIT: 20
int main() {
    int a = 10;
    int b = 20;
    int *p = &a;
    p = p + 1;
    return *p;
}
