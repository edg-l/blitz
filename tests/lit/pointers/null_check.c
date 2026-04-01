// RUN: %tinyc %s -o %t && %t
// EXIT: 1
int main() {
    int *p = (int *)0;
    if (p == (int *)0) { return 1; }
    return 0;
}
