// RUN: %tinyc %s -o %t && %t
// EXIT: 7
int main() {
    unsigned int a = 21;
    unsigned int b = 3;
    return (int)(a / b);
}
