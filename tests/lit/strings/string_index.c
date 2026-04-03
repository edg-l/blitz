// EXIT: 67

// indexing into a string literal (67 = 'C')
int main() {
    char* s = "ABCDE";
    return *(s + 2);
}
