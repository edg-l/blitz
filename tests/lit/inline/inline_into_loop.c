// RUN: %tinyc %s -o %t && %t
// EXIT: 15
// Inline a small function called inside a while loop.
int double_it(int x) { return x + x; }
int main() {
    int sum = 0;
    int i = 1;
    while (i < 6) {
        sum = sum + double_it(i);
        i = i + 1;
    }
    return sum - 15;
}
