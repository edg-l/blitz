int main() {
    char c = 65;
    short s = (short)c;
    int i = (int)s;
    long l = (long)i;
    int result = (int)(sizeof(char) + sizeof(short) + sizeof(int) + sizeof(long));
    return result;
}
