// RUN: %tinyc %s -o %t && %t
// EXIT: 200
int main() {
    short a = 100;
    short b = 100;
    int result = a + b;
    return result;
}
