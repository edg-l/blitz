// RUN: %tinyc %s -o %t && %t
// EXIT: 99
int g;

__attribute__((noinline))
void writer() {
    g = 99;
}

__attribute__((noinline))
int reader() {
    return g;
}

int main() {
    writer();
    return reader();
}
