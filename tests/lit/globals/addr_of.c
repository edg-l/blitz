// RUN: %tinyc %s -o %t && %t
// EXIT: 55

int g;

__attribute__((noinline))
int read_ptr(int *p) {
    return *p;
}

int main() {
    g = 55;
    int *p = &g;
    return read_ptr(p);
}
