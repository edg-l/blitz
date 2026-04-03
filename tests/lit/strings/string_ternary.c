// EXIT: 0

// ternary selecting between two string literals
extern int puts(char* s);
int main() {
    int x = 10;
    char* msg = x > 5 ? "yes" : "no";
    puts(msg);
    // verify the other branch works too
    char* msg2 = x < 5 ? "yes" : "no";
    puts(msg2);
    return 0;
}
