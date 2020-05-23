int main() {

  int a = 5;
  int c = 1;
  while (c < a) {
    c = c+c;
  }
  a = c-a;
  c = 0;

  return 0;
}
