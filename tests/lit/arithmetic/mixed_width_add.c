// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int main() {
    char c = 10;
    short s = 12;
    int i = 20;
    return c + s + i;
}
