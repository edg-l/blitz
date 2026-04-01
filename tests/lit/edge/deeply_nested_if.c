// RUN: %tinyc %s -o %t && %t
// EXIT: 5
int main() {
    int x = 5;
    if (x > 0) {
        if (x > 1) {
            if (x > 2) {
                if (x > 3) {
                    if (x > 4) {
                        return 5;
                    }
                }
            }
        }
    }
    return 0;
}
