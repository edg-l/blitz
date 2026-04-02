// RUN: %tinyc %s -o %t && %t
// EXIT: 81

int main() {
    // sizeof(int[10]) = 10 * 4 = 40
    int s1 = (int)sizeof(int[10]);
    // sizeof(long[2]) = 2 * 8 = 16
    int s2 = (int)sizeof(long[2]);
    // sizeof(char[5]) = 5 * 1 = 5
    int s3 = (int)sizeof(char[5]);
    // sizeof(int[2][3]) = 2 * 3 * 4 = 24
    int s4 = (int)sizeof(int[2][3]);

    // 40 + 16 + 5 + 24 = 85 -- but we check specific values via nested ifs
    // to avoid a sequential-if SSA issue
    if (s1 == 40) {
        if (s2 == 16) {
            if (s3 == 5) {
                if (s4 == 24) {
                    return 81;
                }
            }
        }
    }
    return 0;
}
