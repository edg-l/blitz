// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int main() {
    short s = 42;
    long l = (long)s;
    return (int)l;
}
