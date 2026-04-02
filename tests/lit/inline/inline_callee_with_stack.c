// RUN: %tinyc %s -o %t && %t
// EXIT: 10
// Inline a function that uses a stack slot (addr-taken local).
int use_stack(int x) {
    int local = x;
    int *p = &local;
    return *p + *p;
}
int main() { return use_stack(5); }
