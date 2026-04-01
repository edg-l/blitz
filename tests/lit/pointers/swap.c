// RUN: %tinyc %s -o %t && %t
// EXIT: 21
void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}
int main() {
    int x = 1;
    int y = 2;
    swap(&x, &y);
    return x * 10 + y;
}
