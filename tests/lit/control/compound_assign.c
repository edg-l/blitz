// EXIT: 0
// Test compound assignment operators (+=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=)

int main() {
    int x = 10;
    x += 5;
    if (x != 15) { return 1; }

    x -= 3;
    if (x != 12) { return 2; }

    x *= 2;
    if (x != 24) { return 3; }

    x /= 4;
    if (x != 6) { return 4; }

    x %= 4;
    if (x != 2) { return 5; }

    // Bitwise compound assignment
    int y = 0xFF;
    y &= 0x0F;
    if (y != 0x0F) { return 6; }

    y |= 0xF0;
    if (y != 0xFF) { return 7; }

    y ^= 0x0F;
    if (y != 0xF0) { return 8; }

    // Shift compound assignment
    int z = 1;
    z <<= 4;
    if (z != 16) { return 9; }

    z >>= 2;
    if (z != 4) { return 10; }

    // Compound assign with pointer dereference
    int val = 100;
    int *p = &val;
    *p += 50;
    if (val != 150) { return 11; }

    *p -= 25;
    if (val != 125) { return 12; }

    // Compound assign in loop
    int sum = 0;
    for (int i = 1; i <= 5; i++) {
        sum += i;
    }
    if (sum != 15) { return 13; }

    return 0;
}
