// RUN: %tinyc %s -o %t && %t
// EXIT: 12

int main() {
    int matrix[2][3];
    int i = 0;
    while (i < 2) {
        int j = 0;
        while (j < 3) {
            matrix[i][j] = i * 10 + j;
            j = j + 1;
        }
        i = i + 1;
    }
    // matrix[1][2] = 1*10 + 2 = 12
    return matrix[1][2];
}
