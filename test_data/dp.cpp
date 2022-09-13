#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

void tf_dp (int n, double *y, double lam, double *beta);

void result_write(int n, double *beta, double lam, char *output_filename);

int main(int argc, char *argv[]) {
  double y_val;
  double *y, *beta;
  char *input = argv[1];
  double lam = atof(argv[2]);
  char *output = argv[3];
  FILE *fp = fopen(input, "r");

  int n = 0;
  while (fscanf(fp, "%lf", &y_val) != EOF) {
    n++;
  }
  fclose(fp);
  y = new double[n];
  beta = new double[n];

  int i = 0;
  fp = fopen(input, "r");
  while (fscanf(fp, "%lf", &y_val) != EOF) {
    y[i] = y_val;
    i++;
  }
  fclose(fp);

  for (int i = 0; i < 1000; i++) {
    tf_dp(n, y, lam, beta);
  }

  result_write(n, beta, lam, output);

  return 0;
}

void tf_dp (int n, double *y, double lam, double *beta)
{
  int i;
  int k;
  int l;
  int r;
  int lo;
  int hi;
  double afirst;
  double alast;
  double bfirst;
  double blast;
  double alo;
  double blo;
  double ahi;
  double bhi;
  double *x;
  double *a;
  double *b;
  double *tm;
  double *tp;

  /* Take care of a few trivial cases */
  if (n==0) return;
  if (n==1 || lam==0)
  {
    for (i=0; i<n; i++) beta[i] = y[i];
    return;
  }

  x = (double*) malloc(2*n*sizeof(double));
  a = (double*) malloc(2*n*sizeof(double));
  b = (double*) malloc(2*n*sizeof(double));

  /* These are the knots of the back-pointers */
  tm = (double*) malloc((n-1)*sizeof(double));
  tp = (double*) malloc((n-1)*sizeof(double));

  /* We step through the first iteration manually */
  tm[0] = -lam+y[0];
  tp[0] = lam+y[0];
  l = n-1;
  r = n;
  x[l] = tm[0];
  x[r] = tp[0];
  a[l] = 1;
  b[l] = -y[0]+lam;
  a[r] = -1;
  b[r] = y[0]+lam;
  afirst = 1;
  bfirst = -y[1]-lam;
  alast = -1;
  blast = y[1]-lam;

  /* Now iterations 2 through n-1 */
  for (k=1; k<n-1; k++)
  {
    /* Compute lo: step up from l until the
       derivative is greater than -lam */
    alo = afirst;
    blo = bfirst;
    for (lo=l; lo<=r; lo++)
    {
      if (alo*x[lo]+blo > -lam) break;
      alo += a[lo];
      blo += b[lo];
    }

    /* Compute hi: step down from r until the
       derivative is less than lam */
    ahi = alast;
    bhi = blast;
    for (hi=r; hi>=lo; hi--)
    {
      if (-ahi*x[hi]-bhi < lam) break;
      ahi += a[hi];
      bhi += b[hi];
    }

    /* Compute the negative knot */
    tm[k] = (-lam-blo)/alo;
    l = lo-1;
    x[l] = tm[k];

    /* Compute the positive knot */
    tp[k] = (lam+bhi)/(-ahi);
    r = hi+1;
    x[r] = tp[k];

    /* Update a and b */
    a[l] = alo;
    b[l] = blo+lam;
    a[r] = ahi;
    b[r] = bhi+lam;
    afirst = 1;
    bfirst = -y[k+1]-lam;
    alast = -1;
    blast = y[k+1]-lam;
  }

  /* Compute the last coefficient: this is where
     the function has zero derivative */
  alo = afirst;
  blo = bfirst;
  for (lo=l; lo<=r; lo++)
  {
    if (alo*x[lo]+blo > 0) break;
    alo += a[lo];
    blo += b[lo];
  }
  beta[n-1] = -blo/alo;

  /* Compute the rest of the coefficients, by the
     back-pointers */
  for (k=n-2; k>=0; k--)
  {
    if (beta[k+1]>tp[k]) beta[k] = tp[k];
    else if (beta[k+1]<tm[k]) beta[k] = tm[k];
    else beta[k] = beta[k+1];
  }

  /* Done! Free up memory */
  free(x);
  free(a);
  free(b);
  free(tm);
  free(tp);
}

void result_write(int n, double *beta, double lam, char *output_filename) {
  FILE *fp = fopen(output_filename, "w");
  for(int i = 0; i < n; i++) {
    fprintf(fp, "%lf\n", beta[i]);
  }
  fclose(fp);
}