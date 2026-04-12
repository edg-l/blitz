// Test: f32 pointer dereference
// EXIT: 0

int main() {
    float x = 2.5f;
    float *p = &x;
    float y = *p;
    if (y > 2.0f) {
        return 0;
    }
    return 1;
}
