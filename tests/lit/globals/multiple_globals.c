// RUN: %tinyc %s -o %t && %t
// EXIT: 30
int a;
int b = 10;
long c;
int main() {
    a = 5;
    c = 15;
    return a + b + (int)c;
}
