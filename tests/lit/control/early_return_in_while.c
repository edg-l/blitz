// EXIT: 0
// Early return from inside a while loop.
int find_ge(int *arr, int len, int target) {
    int i = 0;
    while (i < len) {
        if (*(arr + i) >= target) { return i; }
        i = i + 1;
    }
    return 0 - 1;
}
int main() {
    int a = 10;
    int b = 20;
    int c = 30;
    int d = 40;
    if (find_ge(&a, 4, 25) != 2) { return 1; }
    return 0;
}
