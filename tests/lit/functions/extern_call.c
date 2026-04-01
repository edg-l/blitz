// RUN: %tinyc %s -o %t && %t
// EXIT: 21
extern int abs(int x);
int main() {
    return abs(-21);
}
