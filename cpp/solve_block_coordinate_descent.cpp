#include "flsa_dp.hpp"
#include "solve_block_coordinate_descent.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

const double L = 0.5;
const double EPS = 1e-7;

double sigmoid(double x) {
    return 1 / (exp(-x) + 1);
}

double clm_derivative(size_t q, double x, double *b, size_t y) {
    if (y == 1){
        return 1 - sigmoid(b[0] - x);
    }
    if (y == q){
        return - sigmoid(b[q - 1] - x);
    }
    return 1 - sigmoid(b[y - 2] - x) - sigmoid(b[y - 1] - x);
}

void solve_square(size_t n, size_t q, double *f, double *b, double *y, double *fj) {
    double l_inv = 1/L;
    double *solver_y;
    solver_y = new double[n];
    // sort

    // calc solver_y
    for (size_t i = 0; i < n; i++){
        solver_y[i] = l_inv * clm_derivative(q, f[i], b, y[i]);
    }
    return;
}

void set_argsort(vector<vector<size_t>> &argsort, vector<vector<bool>> &argsort_c, vector<vector<size_t>> &argsort_inv, vector<vector<double>> const &x){
    size_t d = x.size();
    size_t n = x[0].size();
    size_t i;
    for (size_t j = 0; j<d; j++){
        vector<double> xj = x[j];
        argsort[j];
        iota(argsort[j].begin(), argsort[j].end(), 0);
        sort(argsort[j].begin(), argsort[j].end(),
            [&xj](size_t left, size_t right) -> bool {
                  // sort indices according to corresponding array element
                  return xj[left] < xj[right];
              });
        for (i = 0; i < n; i++) {
            if (i!=n-1 && xj[argsort[j][i]]==xj[argsort[j][i + 1]]){
                argsort_c[j][i] = 1;
            }
            argsort_inv[j][argsort[j][i]] = i;
        }
    }
    return;

}

void solve_block_coordinate_descent(vector< vector<double>> x, vector< vector<double>>f, vector<double> y , size_t q, double lam) {
    double b[] = {-1, 0, 1};
    size_t n = y.size();
    size_t d = x.size();
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<size_t> > argsort(d, vector<size_t>(n));
    vector<vector<bool> > argsort_c(d, vector<bool>(n-1, 0));
    vector<vector<size_t> > argsort_inv(d, vector<size_t>(n));

    set_argsort(argsort, argsort_c, argsort_inv, x);

    while (1) {
        for (size_t j = 0; j < d; j++){
            for (size_t i = 0; i < n; i++){
                fsumtmp[i] = fsum[i] - f[j][i];
            }
            solve_square(n, q, &fsum[0], b, &y[0], &f[j][0]);
            for (size_t i = 0; i < n; i++){
                fsum[i] = fsumtmp[i] + f[j][i];
            }
        }
    }
}
