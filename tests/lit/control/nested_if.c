// RUN: %tinyc %s -o %t && %t
// EXIT: 3
int main() {
    int x = 15;
    if (x > 10) {
        if (x < 20) { return 3; }
        return 2;
    }
    return 1;
}
