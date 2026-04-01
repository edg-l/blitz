// EXIT: 0
// Signed right shift should use arithmetic shift (sar), preserving sign bit.
int main() {
    int neg8 = 0 - 8;
    int result = neg8 >> 1;
    if (result != 0 - 4) { return 1; }
    int neg1 = 0 - 1;
    if ((neg1 >> 1) != 0 - 1) { return 2; }
    if ((neg1 >> 31) != 0 - 1) { return 3; }
    return 0;
}
