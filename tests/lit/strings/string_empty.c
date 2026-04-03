// EXIT: 0

// empty string literal (just a null terminator in .rodata)
extern int puts(char* s);
int main() {
    puts("");
    return 0;
}
