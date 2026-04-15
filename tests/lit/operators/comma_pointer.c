// RUN: %tinyc %s -o %t && %t
// EXIT: 0

// Test comma with pointers

int main() {
    int a = 10;
    int b = 20;
    int* p = &a;
    int* q = &b;
    
    // Comma with pointer dereference
    int x = (*p, b);
    if (x != 20) {
        return 1;
    }
    
    // Comma with pointers, returns second pointer
    int* r = (p, q);
    if (*r != 20) {
        return 2;
    }
    
    return 0;
}
