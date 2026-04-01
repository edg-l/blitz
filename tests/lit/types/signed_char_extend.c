// RUN: %tinyc %s -o %t && %t
// EXIT: 1
int main() {
    char c = -1;
    int i = (int)c;
    return i + 2;
}
