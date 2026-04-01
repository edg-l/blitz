int main() {
    int x = 170;
    int y = 85;
    int z = (x & y) | ((x ^ y) >> 1);
    return z & 255;
}
