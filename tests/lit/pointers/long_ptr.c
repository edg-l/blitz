// RUN: %tinyc %s -o %t && %t
// EXIT: 99
int main() {
    long v = 99;
    long *p = &v;
    return (int)(*p);
}
