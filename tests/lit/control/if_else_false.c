// RUN: %tinyc %s -o %t && %t
// EXIT: 2
int main() {
    int x = 3;
    if (x > 5) { return 1; } else { return 2; }
}
