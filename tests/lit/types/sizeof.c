// RUN: %tinyc %s -o %t && %t
// EXIT: 15
int main() {
    return sizeof(char) + sizeof(short) + sizeof(int) + sizeof(long);
}
