// RUN: %tinyc %s -o %t && %t
// EXIT: 42
void noop() { return; }
int main() {
    noop();
    return 42;
}
