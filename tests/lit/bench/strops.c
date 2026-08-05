// Hand-written strlen, strcmp, strcpy and reverse over char buffers.
//
// Byte-width loads and stores with a data-dependent exit test. The 8-bit
// operand width is its own lowering path, and `s[i]` against `d[i]` is the
// two-buffer aliasing question offset disjointness cannot answer.

// OUTPUT: 60
// OUTPUT: 0
// OUTPUT: -9
// OUTPUT: 253407
// EXIT: 0


extern int printf(char* fmt, int x);

int my_strlen(char* s) {
    int n = 0;
    while (s[n] != 0) {
        n = n + 1;
    }
    return n;
}

int my_strcmp(char* a, char* b) {
    int i = 0;
    while (a[i] != 0 && a[i] == b[i]) {
        i = i + 1;
    }
    return (int)a[i] - (int)b[i];
}

void my_strcpy(char* dst, char* src) {
    int i = 0;
    while (src[i] != 0) {
        dst[i] = src[i];
        i = i + 1;
    }
    dst[i] = 0;
}

void reverse(char* s, int n) {
    int i = 0;
    int j = n - 1;
    while (i < j) {
        char t = s[i];
        s[i] = s[j];
        s[j] = t;
        i = i + 1;
        j = j - 1;
    }
}

int main() {
    char src[64];
    char dst[64];

    // "abcdefghij" repeated to 60 characters.
    for (int i = 0; i < 60; i = i + 1) {
        src[i] = (char)(97 + i % 10);
    }
    src[60] = 0;

    int len = my_strlen(src);
    my_strcpy(dst, src);
    int same = my_strcmp(src, dst);

    reverse(dst, len);
    int after = my_strcmp(src, dst);

    int fold = 0;
    for (int i = 0; i < len; i = i + 1) {
        fold = fold * 31 + (int)dst[i];
        fold = fold % 1000003;
    }

    printf("%d\n", len);
    printf("%d\n", same);
    printf("%d\n", after);
    printf("%d\n", fold);
    return 0;
}
