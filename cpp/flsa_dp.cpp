void tf_dp(int n, double *y, double lam, bool *c, double *beta) {
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
  if (n == 0) return;
  if (n == 1 || lam == 0) {
    for (i = 0; i < n; i++) beta[i] = y[i];
    return;
  }

  x = new double[2 * n];
  a = new double[2 * n];
  b = new double[2 * n];

  /* These are the knots of the back-pointers */
  tm = new double[n - 1];
  tp = new double[n - 1];

  /* We step through the first iteration manually */

  /*
  tm, tp:ψ_kの範囲が[tm, tp]
  */
  tm[0] = -lam + y[0];
  tp[0] = lam + y[0];

  /*
  論文中では使わなくなったknotをdeleteしているが,
  ここではlとrを移動させることで対処している．
  kが増えて行ってもknotのx座標，傾きの変化は継承されていく
  δ_k'において採用するknotが変化していき，　それをlとrによって表現している．
  l-r+1: num of knots
  l,rは[0,2n]の範囲で自由に動く可能性がある.
  */
  l = n - 1;
  r = n;

  /*
  a, b: その区間の直線の方程式は　δ'(y)= a*y+b
  x: δのノットのx座標
  b = -x*a
  */
  x[l] = tm[0];
  x[r] = tp[0];
  a[l] = 1;
  b[l] = -y[0] + lam;
  a[r] = -1;
  b[r] = y[0] + lam;
  /*
  afirst：二乗誤差では1, e_kの傾きa
  bfirst:δ_kの最初の区間の直線の切片
  */

  afirst = 1;
  bfirst = -y[1] - lam;
  /*
  alast： 二乗誤差では-1, -e_kの傾き
  blast: -δ_kの最初の区間の直線の切片
  */
  alast = -1;
  blast = y[1] - lam;

  /* δ_N'の計算*/
  /* 各イタレーションでδ_k'の計算*/
  /* Now iterations 2 through n-1 */
  for (k = 1; k < n - 1; k++) {
    if(c[i]){
      afirst += 1;
      alast -= y[k + 1];
      bfirst -= 1;
      blast += y[k + 1];
    }
    else{
      /* Compute lo: step up from l until the
        derivative is greater than -lam */

      alo = afirst;
      blo = bfirst;
      for (lo = l; lo <= r; lo++) {
        /*
        alo, blo: その区間の直線の方程式は　δ'(b)= alo*b+blo
        x[lo]: δのノットのx座標
        */
        /*
        δ_k'(b)>-λとなったらbreak
        */
        if (alo * x[lo] + blo > -lam) break;
        alo += a[lo];
        blo += b[lo];
      }

      /* Compute hi: step down from r until the
        derivative is less than lam */
      ahi = alast;
      bhi = blast;
      for (hi = r; hi >= lo; hi--) {
        if (-ahi * x[hi] - bhi < lam) break;
        ahi += a[hi];
        bhi += b[hi];
      }

      /*
        iter毎にδ_k'(b)=-λとなるbについて, knotを追加(id:l, x[l], a[l],
        b[l])をupdate iter毎にδ_k'(b)=λとなるbについて, knotを追加(id:r, x[r],
        a[r], b[r])をupdate
      */
      /* Compute the negative knot */
      tm[k] = (-lam - blo) / alo;
      l = lo - 1;
      x[l] = tm[k];

      /* Compute the positive knot */
      tp[k] = (lam + bhi) / (-ahi);
      r = hi + 1;
      x[r] = tp[k];

      /* Update a and b */
      a[l] = alo;
      b[l] = blo + lam;
      a[r] = ahi;
      b[r] = bhi + lam;

      afirst = 1;
      bfirst = -y[k + 1] - lam;
      alast = -1;
      blast = y[k + 1] - lam;
    }
  }
  /*δ_Nの計算はここまで*/

  /* Compute the last coefficient: this is where
     the function has zero derivative */
  alo = afirst;
  blo = bfirst;
  for (lo = l; lo <= r; lo++) {
    if (alo * x[lo] + blo > 0) break;
    alo += a[lo];
    blo += b[lo];
  }
  beta[n - 1] = -blo / alo;

  /* Compute the rest of the coefficients, by the
     back-pointers */
  for (k = n - 2; k >= 0; k--) {
    if (beta[k + 1] > tp[k])
      beta[k] = tp[k];
    else if (beta[k + 1] < tm[k])
      beta[k] = tm[k];
    else
      beta[k] = beta[k + 1];
  }

  /* Done! Free up memory */
  delete[] x;
  delete[] a;
  delete[] b;
  delete[] tm;
  delete[] tp;
}
