// RUN: %tinyc %s -o %t && %t
// EXIT: 30
int double_val(int x) { return x + x; }
int main() {
    int a = double_val(5);
    int b = double_val(10);
    return a + b;
}
