// Helper file for multi_file_global.c -- not a standalone test.
int counter = 0;

void increment() {
    counter = counter + 1;
}

int get_counter() {
    return counter;
}
