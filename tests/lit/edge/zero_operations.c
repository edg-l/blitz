// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int main() {
    int x = 42;
    int a = x + 0;
    int b = a - 0;
    int c = b * 1;
    int d = c / 1;
    return d;
}
