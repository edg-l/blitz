// EXIT: 0
// Void function that modifies state via pointer. Verify the dummy return
// value doesn't interfere with subsequent computations.
void set(int *p, int val) { *p = val; }
int main() {
    int a = 0;
    int b = 0;
    set(&a, 42);
    set(&b, 99);
    if (a != 42) { return 1; }
    if (b != 99) { return 2; }
    return 0;
}
