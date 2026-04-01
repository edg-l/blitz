// RUN: %tinyc %s -o %t && %t
// EXIT: 65
int main() {
    char c = 65;
    char *p = &c;
    return (int)(*p);
}
