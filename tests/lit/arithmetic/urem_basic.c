// RUN: %tinyc %s -o %t && %t
// EXIT: 2
int main() {
    unsigned int a = 17;
    unsigned int b = 5;
    return (int)(a % b);
}
