// RUN: %tinyc %s -o %t && %t
// EXIT: 255
int main() {
    long big = 4294967295;
    return (int)(big & 255);
}
