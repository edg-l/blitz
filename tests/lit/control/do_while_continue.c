// Test: do-while with continue
// Known bug: e-graph class merging causes incorrect code when body has continue
// EXIT: 12

int main() {
    int sum = 0;
    int i = 0;
    do {
        i = i + 1;
        if (i == 3) {
            continue;
        }
        sum = sum + i;
    } while (i < 5);
    return sum;
}
