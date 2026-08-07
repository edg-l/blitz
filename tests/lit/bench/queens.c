// Eight queens, counting every solution by backtracking.
//
// Deep recursion with three arrays kept live across each recursive call and a
// short scan loop per level -- the shape that puts the most values in flight
// across a call site of anything in this corpus.

// OUTPUT: 92
// EXIT: 0


extern int printf(char* fmt, ...);

int place(int row, int n, int* cols, int* diag1, int* diag2) {
    if (row == n) {
        return 1;
    }
    int found = 0;
    for (int c = 0; c < n; c = c + 1) {
        int d1 = row + c;
        int d2 = row - c + n;
        if (cols[c] == 0 && diag1[d1] == 0 && diag2[d2] == 0) {
            cols[c] = 1;
            diag1[d1] = 1;
            diag2[d2] = 1;
            found = found + place(row + 1, n, cols, diag1, diag2);
            cols[c] = 0;
            diag1[d1] = 0;
            diag2[d2] = 0;
        }
    }
    return found;
}

int main() {
    int cols[8];
    int diag1[16];
    int diag2[17];

    for (int i = 0; i < 8; i = i + 1) {
        cols[i] = 0;
    }
    for (int i = 0; i < 16; i = i + 1) {
        diag1[i] = 0;
    }
    for (int i = 0; i < 17; i = i + 1) {
        diag2[i] = 0;
    }

    int solutions = place(0, 8, cols, diag1, diag2);
    printf("%d\n", solutions);
    return 0;
}
