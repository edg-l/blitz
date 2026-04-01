// RUN: %tinyc %s -o %t && %t
// EXIT: 100
int main() {
    char a = 60;
    char b = 40;
    return (int)(a + b);
}
