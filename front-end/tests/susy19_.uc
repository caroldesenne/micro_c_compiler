int main() {
  
  int v[][] = {{1,2},{3,4},{5,6}};
  int k[3][2];
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 2; j++) {
      k[i][j] = v[i][j];
      print(k[i][j], " ");
    }
    print("     ");
  }
  return 0;

}
