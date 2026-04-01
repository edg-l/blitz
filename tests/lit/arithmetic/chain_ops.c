// RUN: %tinyc %s -o %t && %t
// EXIT: 14
int main() {
    int a = 2;
    int b = 3;
    int c = 4;
    return a * b + c * c / 2;
}
