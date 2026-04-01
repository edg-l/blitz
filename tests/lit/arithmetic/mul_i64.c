// RUN: %tinyc %s -o %t && %t
// EXIT: 6
int main() {
    long a = 1000000000;
    long b = 6;
    long c = a * b;
    return (int)(c / 1000000000);
}
