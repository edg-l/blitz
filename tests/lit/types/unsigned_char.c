// RUN: %tinyc %s -o %t && %t
// EXIT: 200
int main() {
    unsigned char c = 200;
    return (int)c;
}
