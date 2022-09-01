#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  /* random number generator */
  random_device seed;
  mt19937 engine(seed());

  /* initializing generator */
  double mu = 0.0, sig = 1.0;
  normal_distribution<> dist(mu, sig);

  /* generating random number along sin */
  int n = atoi(argv[1]);
  double sample[n];
  for (int i = 0; i < n; i++) {
    sample[i] = sin(i*0.001) + dist(engine);
  }

  /* output */
  FILE *fp = fopen(argv[2], "w");
  for (int i = 0; i < n; i++) {
    fprintf(fp, "%lf\n", sample[i]);
  }
  fclose(fp);
  return 0;
}
