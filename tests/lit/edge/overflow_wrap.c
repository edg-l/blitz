// RUN: %tinyc %s -o %t && %t
// EXIT: 0
int main() {
    int x = 2147483647;
    int y = x + 1;
    return y & 255;
}
