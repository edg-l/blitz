// RUN: %tinyc %s -o %t && %t
// EXIT: 24
int main() {
    int count = 0;
    int i = 0;
    while (i < 2) {
        int j = 0;
        while (j < 3) {
            int k = 0;
            while (k < 4) {
                count = count + 1;
                k = k + 1;
            }
            j = j + 1;
        }
        i = i + 1;
    }
    return count;
}
