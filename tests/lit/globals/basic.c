// RUN: %tinyc %s -o %t && %t
// EXIT: 7
int g;
int main() {
    g = 7;
    return g;
}
