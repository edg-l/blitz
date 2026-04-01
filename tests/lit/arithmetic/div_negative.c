// EXIT: 0
// Signed division with negative operands.
int main() {
    int neg10 = 0 - 10;
    int neg3 = 0 - 3;
    if (neg10 / 3 != 0 - 3) { return 1; }
    if (10 / neg3 != 0 - 3) { return 2; }
    if (neg10 / neg3 != 3) { return 3; }
    if (neg10 % 3 != 0 - 1) { return 4; }
    if (10 % neg3 != 1) { return 5; }
    return 0;
}
