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

    if (!(a != 25) && (b == 5)) {
        i = g(a);
        k = g(2*a);
        a = 50;
    } else {
        i = g(5*a);
        k = g(3*a);
        a = 15;
    }

    return 2*a;
}