// RUN: %tinyc %s -o %t && %t
// EXIT: 3
int main() {
    long a = 1000000000;
    long b = 2000000000;
    long c = a + b;
    return (int)(c / 1000000000);
}
