// Test: do-while with a function call in body and break
// Regression: barrier sort skipped branch-only blocks, clobbering EFLAGS
// Also exercises that calls in the loop body don't perturb the flags chain
// Loop: call inc(i), then check if result >= 4 to break.
// i=0: t=inc(0)=1, 1<4 continue, sum+=1=1, i=1
// i=1: t=inc(1)=2, 2<4 continue, sum+=2=3, i=2
// i=2: t=inc(2)=3, 3<4 continue, sum+=3=6, i=3
// i=3: t=inc(3)=4, 4>=4 break
// EXIT: 6

__attribute__((noinline)) int inc(int x) {
    return x + 1;
}

int main() {
    int sum = 0;
    int i = 0;
    do {
        int t = inc(i);
        i = i + 1;
        if (t >= 4) {
            break;
        }
        sum = sum + t;
    } while (i < 10);
    return sum;
}
