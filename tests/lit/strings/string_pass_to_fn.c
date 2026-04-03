// EXIT: 0

// string literal passed through a function parameter
extern int puts(char* s);
void print_it(char* msg) {
    puts(msg);
}
int main() {
    print_it("passed string");
    return 0;
}
