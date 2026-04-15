// RUN: %tinyc %s -o %t && %t
// EXIT: 0

int identity(int x) {
    return x;
}

int main() {
    int x = (identity(1), identity(2), identity(3), identity(4), identity(5));
    
    if (x != 5) {
        return 1;
    }
    
    int y = ((1, 2), (3, 4));
    
    if (y != 4) {
        return 2;
    }
    
    int z = (1, 2) + (3, 4);
    
    if (z != 6) {
        return 3;
    }
    
    return 0;
}
