// EXIT: 0
// Basic unsigned arithmetic tests.
int main() {
    unsigned a = 10;
    unsigned b = 3;
    if (a / b != 3) { return 1; }
    if (a % b != 1) { return 2; }
    return 0;
}
