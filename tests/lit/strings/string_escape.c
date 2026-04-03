// RUN: %tinyc %s -o %t && %t
// EXIT: 0
extern int puts(char* s);
int main() {
    puts("hello\tworld\n");
    return 0;
}
