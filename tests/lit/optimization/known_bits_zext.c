// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Test known-bits: Zext upper bits are known-zero, masking is redundant.

__attribute__((noinline))
int zext_and_mask(int x) {
    // Cast to char truncates to 8 bits, then widen back.
    // The And with 0xFF should be eliminated since upper bits are already zero.
    char c = (char)x;
    int wide = c;
    return wide & 0xFF;
}

__attribute__((noinline))
int zext_wider_mask(int x) {
    // Mask with 0xFFFF on a value that's already masked to 0xFF.
    int masked = x & 0xFF;
    return masked & 0xFFFF;
}

int main() {
    if (zext_and_mask(42) != 42) { return 1; }
    if (zext_and_mask(0) != 0) { return 2; }
    if (zext_and_mask(255) != 255) { return 3; }
    if (zext_and_mask(256) != 0) { return 4; }

    if (zext_wider_mask(42) != 42) { return 5; }
    if (zext_wider_mask(200) != 200) { return 6; }

    return 0;
}
