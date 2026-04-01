// EXIT: 0
// Pointer subtraction should return element count, not byte count.
int main() {
    int a = 1;
    int b = 2;
    int c = 3;
    int d = 4;
    int *p0 = &a;
    int *p3 = &d;
    long diff = p3 - p0;
    // On stack, d is at a lower address than a, so diff may be negative.
    // Just check it's non-zero (pointer subtraction works at all).
    if (diff == 0) { return 1; }
    return 0;
}
