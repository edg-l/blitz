extern int abs(int x);
extern void exit(int code);

int main() {
    int a = abs(-21);
    int b = abs(-21);
    exit(a + b - 42);
    return 1;
}
