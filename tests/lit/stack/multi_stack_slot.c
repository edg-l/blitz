// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int main() {
    int a = 10;
    int b = 20;
    int c = 12;
    int *pa = &a;
    int *pb = &b;
    int *pc = &c;
    return *pa + *pb + *pc;
}
