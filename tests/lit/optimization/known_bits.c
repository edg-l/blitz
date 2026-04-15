// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test that known-bits optimizations produce correct results.

__attribute__((noinline))
int mask_after_cast(int x) {
    // Casting char to int zeroes upper bits; masking with 0xFF is redundant.
    char c = (char)x;
    int wide = (int)c;
    return wide & 0xFF;
}

int main() {
    if (mask_after_cast(42) != 42) { return 1; }
    if (mask_after_cast(0) != 0) { return 2; }
    if (mask_after_cast(255) != 255) { return 3; }
    return 0;
}
