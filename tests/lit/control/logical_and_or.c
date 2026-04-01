// RUN: %tinyc %s -o %t && %t
// EXIT: 3
int main() {
    int r = 0;
    if (1 && 1) { r = r + 1; }
    if (1 || 0) { r = r + 1; }
    if (!(0 && 1)) { r = r + 1; }
    return r;
}
