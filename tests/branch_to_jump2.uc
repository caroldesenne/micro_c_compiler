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

    while (!(a != 25) && (b == 5)) {
    //while (1 == 2) {
        i = g(10);
        k = g(2*1);
        if (j == 100) {
            i = 100;
            //return 0;
        }
    }

    return 2*a;
}