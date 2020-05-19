int v[][] = {{1,3}, {2,6}, {3,9}};

int main () {
    int k[][] = {{1,2},{0,3}}; 
    int r[3][2];
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 2; j++) {
        r[i][j] = 0;
        for(int m = 0; m < 2; m++) {
          r[i][j] += v[i][m]*k[m][j];
        }
        print(r[i][j]," ");
      }
    }
    return 0;
}
