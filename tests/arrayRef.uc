int main() {

  int v[] = {1,2,3,4};
  int n = 2;
  int a = v[n];
  int b = a;

  int c[2][3];
  int d[3] = c[0];
  int e[] = c[1]; // check size
  int f[3] = c[1];

  char g[4][5];
  char h[4] = g[0];
  char j = h[0];
  char s = h[d[0]];

  return 0;
}
