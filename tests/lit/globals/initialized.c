// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int x = 42;
int main() {
    return x;
}
