int g (int t) {
    int x;
    t *= 2;
    x = 2*t;
    return x;
}

int main() {
    int i, j, k;
    int a = 10;
    int b = 5;

    a = 2*a;
    a = a + b;

    //while (!(a != 25) && (b == 5)) {
    while (g(a) == 0) {
        if (j == 100) {
            i = a;
            g(i);
        }
    }

    i = a;

    g(i);

    return i;
}