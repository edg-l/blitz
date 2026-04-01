// RUN: %tinyc %s -o %t && %t
// EXIT: 1
int main() {
    int x = 10;
    if (x > 5) { return 1; }
    return 0;
}
