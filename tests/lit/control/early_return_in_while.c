// EXIT: 0
// Early return from inside a while loop.
int find_first_nonzero(int *arr, int len) {
    int i = 0;
    while (i < len) {
        if (*(arr + i) != 0) { return i; }
        i = i + 1;
    }
    return 0 - 1;
}
int main() {
    int arr = 0;
    int *p = &arr;
    // Store values: [0, 0, 42, 0] via pointer offsets
    *(p + 0) = 0;
    *(p + 1) = 0;
    *(p + 2) = 42;
    *(p + 3) = 0;
    if (find_first_nonzero(p, 4) != 2) { return 1; }
    if (find_first_nonzero(p, 1) != 0 - 1) { return 2; }
    return 0;
}
