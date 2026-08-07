// Bitwise CRC-32 over a 512-byte buffer, no lookup table.
//
// Eight shift-and-conditional-xor steps per byte, all on `unsigned int`: a
// register-resident loop with almost no memory traffic, where the whole cost is
// the arithmetic the e-graph is meant to fold.

// OUTPUT: 1567775976
// EXIT: 0


extern int printf(char* fmt, ...);

unsigned int crc32_byte(unsigned int crc, unsigned int byte) {
    crc = crc ^ byte;
    for (int k = 0; k < 8; k = k + 1) {
        if (crc & 1) {
            crc = (crc >> 1) ^ 3988292384;
        } else {
            crc = crc >> 1;
        }
    }
    return crc;
}

int main() {
    unsigned char buf[512];
    unsigned int seed = 2463534242;
    for (int i = 0; i < 512; i = i + 1) {
        seed = seed ^ (seed << 13);
        seed = seed ^ (seed >> 17);
        seed = seed ^ (seed << 5);
        buf[i] = (unsigned char)(seed & 255);
    }

    unsigned int crc = 4294967295;
    for (int i = 0; i < 512; i = i + 1) {
        crc = crc32_byte(crc, (unsigned int)buf[i]);
    }
    crc = crc ^ 4294967295;

    printf("%u\n", crc);
    return 0;
}
