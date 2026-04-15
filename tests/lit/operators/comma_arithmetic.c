// RUN: %tinyc %s -o %t && %t
// EXIT: 0

// Test comma result type flows correctly for arithmetic

int main() {
    // Comma result used in multiplication
    int x = (1, 2) * 3;
    if (x != 6) {
        return 1;
    }
    
    // Comma result used in division
    int y = (10, 20) / 4;
    if (y != 5) {
        return 2;
    }
    
    // Chained comma with arithmetic
    int z = (1, 2, 3) + (4, 5, 6);
    if (z != 9) {
        return 3;
    }
    
    // Comma in arithmetic expression
    int result = (1, 2) + (3, 4);
    if (result != 6) {
        return 4;
    }
    
    return 0;
}
