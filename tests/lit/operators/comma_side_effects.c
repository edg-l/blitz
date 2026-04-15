// RUN: %tinyc %s -o %t && %t
// EXIT: 0

int add_one(int x) {
    return x + 1;
}

int main() {
    int a = 1;
    int b = 2;
    
    int result = (add_one(a), add_one(b), a + b);
    
    if (a != 1) {
        return 1;
    }
    if (b != 2) {
        return 2;
    }
    if (result != 3) {
        return 3;
    }
    
    return 0;
}
