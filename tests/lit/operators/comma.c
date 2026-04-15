// RUN: %tinyc %s -o %t && %t
// EXIT: 0

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = (add(1, 2), add(3, 4));
    int x = (1, 2, 3, 4, 5);
    int total = result + x;
    
    if (result != 7) {
        return 1;
    }
    if (x != 5) {
        return 2;
    }
    if (total != 12) {
        return 3;
    }
    
    return 0;
}
