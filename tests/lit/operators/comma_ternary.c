// RUN: %tinyc %s -o %t && %t
// EXIT: 0

// Test comma with ternary operator

int main() {
    int cond = 1;
    
    // Comma in both branches of ternary
    int x = cond ? (1, 2) : (3, 4);
    if (x != 2) {
        return 1;
    }
    
    cond = 0;
    int y = cond ? (1, 2) : (3, 4);
    if (y != 4) {
        return 2;
    }
    
    // Ternary as left operand of comma
    int z = (cond ? 1 : 2, 5);
    if (z != 5) {
        return 3;
    }
    
    return 0;
}
