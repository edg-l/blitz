// RUN: %tinyc %s -o %t && %t
// EXIT: 0
int main() {
    int x = 10;
    while (x > 0) { x = x - 1; }
    return x;
}
