// EXIT: 0
// Sign extension and truncation edge cases.
int main() {
    int a = 200;
    char b = (char)a;
    long c = (long)b;
    if (c != 0 - 56) { return 1; }

    int d = 42;
    long e = (long)(char)d;
    if (e != 42) { return 2; }

    return 0;
}
