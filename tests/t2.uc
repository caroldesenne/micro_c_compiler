int n = 3;

int doubleMe (int x) {
    return x * x;
}

int main () {
    int v = n;
    v = doubleMe (v);
    assert v == n * n;
    return 0;
}