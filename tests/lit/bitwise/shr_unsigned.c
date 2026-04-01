// RUN: %tinyc %s -o %t && %t
// EXIT: 127
int main() {
    unsigned int x = 254;
    return (int)(x >> 1);
}
