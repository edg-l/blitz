// EXIT: 0
// Early return from inside a while loop.
// Uses a single counter-based approach to avoid pointer arithmetic past allocation.
int count_up_to(int limit) {
    int i = 0;
    while (i < limit) {
        if (i + i == 4) { return i; }
        i = i + 1;
    }
    return 0 - 1;
}
int main() {
    if (count_up_to(10) != 2) { return 1; }
    if (count_up_to(1) != 0 - 1) { return 2; }
    return 0;
}
