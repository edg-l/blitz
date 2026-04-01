// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int main() {
    int i = 42;
    long l = (long)i;
    return (int)l;
}
