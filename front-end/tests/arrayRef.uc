int main() {

  int v[] = {1,2,3,4};
  int n = 2;
  int a = v[n];
  int b = a;

  int c[2][3];
  int d[3] = c[0];
  int e[] = c[1]; // check size
  int f[3] = c[1];

  char g[3][4][5];
  char h[4][5] = g[0];
  char i[5] = h[2];
  char j[] = h[0];
  char k = h[0][1];
}
