// RUN: %tinyc %s -o %t && %t
// EXIT: 6
int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a - (a / b) * b;
        a = t;
    }
    return a;
}
int main() { return gcd(54, 24); }
