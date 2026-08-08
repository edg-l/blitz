// Helper file for cross_file_uncalled_definition.c -- not a standalone test.
int helper(int x);

int forward(int x) {
    return helper(x) + 1;
}
