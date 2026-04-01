// RUN: %tinyc %s -o %t && %t
// EXIT: 12
int main() {
    int count = 0;
    int i = 0;
    while (i < 3) {
        int j = 0;
        while (j < 4) {
            count = count + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return count;
}
