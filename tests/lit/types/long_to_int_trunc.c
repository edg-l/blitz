// RUN: %tinyc %s -o %t && %t
// EXIT: 100
int main() {
    long l = 4294967396;
    return (int)l;
}
