// RUN: %tinyc %s --enable-licm -o %t && %t
// OUTPUT: pass
// EXIT: 0

// End-to-end correctness test: compute(3, 4) should return (3+4)*10 = 70.

extern int puts(char* s);

int compute(int a, int b) {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        int z = a + b;
        sum = sum + z;
        i = i + 1;
    }
    return sum;
}

int main() {
    int result = compute(3, 4);
    if (result == 70) {
        puts("pass");
    }
    return 0;
}
