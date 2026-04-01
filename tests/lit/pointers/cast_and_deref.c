// EXIT: 0
// Cast int pointer to char pointer and dereference.
int main() {
    int val = 513;
    char *cp = (char*)&val;
    // On little-endian x86: 513 = 0x0201, first byte is 1.
    int first_byte = (int)(unsigned char)*cp;
    if (first_byte != 1) { return 1; }
    return 0;
}
