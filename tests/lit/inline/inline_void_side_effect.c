// RUN: %tinyc %s -o %t && %t
// EXIT: 42
// Inline a void function that writes through a pointer.
void store(int *p, int v) { *p = v; }
int main() {
    int x = 0;
    store(&x, 42);
    return x;
}
