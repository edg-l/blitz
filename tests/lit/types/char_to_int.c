// RUN: %tinyc %s -o %t && %t
// EXIT: 127
int main() {
    char c = 127;
    return (int)c;
}
