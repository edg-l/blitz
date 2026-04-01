// RUN: %tinyc %s -o %t && %t
// EXIT: 21
int main() {
    int a = 1;
    int b = 2;
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
    return a * 10 + b;
}
