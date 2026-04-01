// RUN: %tinyc %s -o %t && %t
// EXIT: 3
int main() {
    int r = 0;
    if (5 <= 5) { r = r + 1; }
    if (5 >= 5) { r = r + 1; }
    if (5 <= 10) { r = r + 1; }
    return r;
}
