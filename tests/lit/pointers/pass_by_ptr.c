// RUN: %tinyc %s -o %t && %t
// EXIT: 42
void set(int *p, int v) {
    *p = v;
}
int main() {
    int x = 0;
    set(&x, 42);
    return x;
}
