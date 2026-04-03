// EXIT: 7

// break and continue in while loops
int main() {
    int i = 0;
    int sum = 0;
    while (i < 100) {
        if (i == 5) {
            break;
        }
        sum = sum + i;
        i = i + 1;
    }
    // 0+1+2+3+4 = 10, but i == 5 at exit
    // return i + sum%3 to keep it under 255
    return sum + i - 8;
}
