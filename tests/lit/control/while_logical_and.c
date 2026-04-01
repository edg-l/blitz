// EXIT: 0
// While loop with && in condition.
int main() {
    int x = 10;
    int y = 0;
    while (x > 0 && y < 5) {
        y = y + 1;
        x = x - 1;
    }
    if (y != 5) { return 1; }
    if (x != 5) { return 2; }
    return 0;
}
