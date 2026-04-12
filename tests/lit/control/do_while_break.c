// Test: do-while with break
// Known bug: e-graph class merging causes incorrect code when body has break
// EXIT: 3

int main() {
    int sum = 0;
    int i = 0;
    do {
        i = i + 1;
        if (i == 3) {
            break;
        }
        sum = sum + i;
    } while (i < 10);
    return sum;
}
